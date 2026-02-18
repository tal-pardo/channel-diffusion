import matplotlib.pyplot as plt
import json
import argparse
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset

def get_config_pnp(config_main, config_car, config_ant, config_diff, config_noise):
    """
    Load five explicit config file paths for PnP use.
    Returns: configs, configs_car, configs_ant, configs_diff, configs_noise
    """
    with open(config_main, 'r') as f:
        configs = argparse.Namespace(**json.load(f))
    with open(config_car, 'r') as f:
        configs_car = argparse.Namespace(**json.load(f))
    with open(config_ant, 'r') as f:
        configs_ant = argparse.Namespace(**json.load(f))
    with open(config_diff, 'r') as f:
        configs_diff = argparse.Namespace(**json.load(f))
    with open(config_noise, 'r') as f:
        configs_noise = argparse.Namespace(**json.load(f))

    configs.BS_ant = torch.tensor(configs.BS_ant)
    if torch.cuda.is_available():
        configs.BS_ant = configs.BS_ant.cuda()

    return configs, configs_car, configs_ant, configs_diff, configs_noise


def load_dataset_deepmimo(args):
# def get_dataset(ratio: list, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='.'):
    dest_train = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_train.pt")
    dest_test = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_test.pt")
    dest_val = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_val.pt")  # not always exist

    if os.path.exists(dest_train) and os.path.exists(dest_test):
        channel_train = torch.load(dest_train)
        channel_test = torch.load(dest_test)
        if os.path.exists(dest_val):
            channel_val = torch.load(dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]
    else:
        channel_file = os.path.join(args.dataset_dir, args.dataset_file_name, "data.npy")
        print("Loading channels", channel_file)

        channel = np.load(channel_file)  # N*1*ant*car complex
        channel_torch = torch.tensor(channel).squeeze(1)  # N*ant*car cfloat tensor
        num_data = channel_torch.shape[0]

        # Normalization
        channel_torch = channel_torch * 1e5

        perm = torch.randperm(num_data)

        num = int(args.ratio[0] * num_data)
        ids = perm[0:num]
        channel_train = channel_torch[ids]
        torch.save(channel_train, dest_train)

        num2 = int(args.ratio[1] * num_data)
        ids = perm[num:num+num2]
        channel_test = channel_torch[ids]
        torch.save(channel_test,  dest_test)

        if len(args.ratio) == 3:
            num3 = int(args.ratio[2] * num_data)
            ids = perm[num+num2:num+num2+num3]
            channel_val = channel_torch[ids]
            torch.save(channel_val,  dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]

def load_dataset(args):
    if args.dataset == "deepmimo":
        return load_dataset_deepmimo(args)
    else:
        raise NotImplementedError

# LS estimator for comparison
class LeastSquaresSampler:
    """
    Standard Least Squares (LS) Estimator with Linear Interpolation.
    Acts as a baseline for comparison.
    """
    def __init__(self, config, config_noise, device):
        self.config = config
        self.config_noise = config_noise
        self.device = device
        self.num_car = config.num_car
        self.num_ant = config.num_ant 
        from pnp_posterior_sampling.pnp_sampling_v2 import PnP_SignalGenerator      
        self.pnp_signal_generator = PnP_SignalGenerator(self.config, self.config_noise, self.device)
        self.pilot_mask = self.pnp_signal_generator.pilot_mask  		# shape: (num_car,) {0,1}
        self.data_mask = 1.0 - self.pilot_mask							# shape: (num_car,) {0,1}
        self.pilots = (1.0+0.0j) * self.pilot_mask		            	# shape: (num_car,) {complex tensor with 1 in pilot positions and 0 in data positions}	
        self.num_pilot = len(self.config.pilot_cars)
        self.sigma_n = self.config_noise.noise_power
        self.pilot_indices = self.config.pilot_cars

        # Create interpolation grid (0 to 63)
        self.grid_indices = torch.arange(self.num_car, device=device).float()

    def estimate(self, Y):
        H_est = torch.zeros_like(Y, dtype=torch.complex64)
        
        # LS at Pilot positions
        pilot_idx_tensor = torch.tensor(self.pilot_indices, device=self.device)
        H_ls_pilots = Y[:, :, pilot_idx_tensor] / (1.0 + 0.0j) 
        
        # Place them in the final matrix
        H_est[:, :, pilot_idx_tensor] = H_ls_pilots
        
        # Linear Interpolation between pilots
        for i in range(len(self.pilot_indices) - 1):
            start_idx = self.pilot_indices[i]
            end_idx = self.pilot_indices[i+1]
            
            # Use i and i+1 to get the small extracted pilot values
            val_start = H_ls_pilots[:, :, i].unsqueeze(-1)
            val_end = H_ls_pilots[:, :, i+1].unsqueeze(-1)
            
            dist = end_idx - start_idx
            steps = torch.arange(1, dist, device=self.device).view(1, 1, -1) / float(dist)
            
            interp_vals = (1 - steps) * val_start + steps * val_end
            H_est[:, :, start_idx+1:end_idx] = interp_vals

        # 3. Extrapolation for edges
        if self.pilot_indices[0] > 0:
            H_est[:, :, :self.pilot_indices[0]] = H_ls_pilots[:, :, 0].unsqueeze(-1)
        if self.pilot_indices[-1] < self.num_car - 1:
            H_est[:, :, self.pilot_indices[-1]+1:] = H_ls_pilots[:, :, -1].unsqueeze(-1)
            
        return H_est

# --- Plotting utility for PnP and LS results ---
def my_plots():
    """
    Plots NMSE, BER, and Phase Shift for PnP sampling and Least Squares results.
    Uses the results from the attached table (hardcoded for now).
    """
    # SNR values (dB)
    snr_db = [-3.5, 0, 3.5, 5, 10]
    # PnP results
    pnp_nmse = [0.705, 0.705, 0.397, 0.252, 0.217]
    pnp_ber = [0.0514, 0.0512, 0.0466, 0.0447, 0.0458]
    pnp_ps = [3.09, 3.19, 3.13, 3.23, 3.19]
    # LS results
    ls_nmse = [2.295, 1.767, 1.530, 1.475, 1.382]
    ls_ber = [0.416, 0.417, 0.417, 0.416, 0.415]
    ls_ps = [-43.64, -46.28, -47.90, -47.98, -48.77]

    # NMSE plot
    plt.figure(figsize=(6,4))
    plt.plot(snr_db, pnp_nmse, marker='o', label='PnP sampling')
    plt.plot(snr_db, ls_nmse, marker='s', label='Least Squares')
    plt.xlabel('SNR [dB]')
    plt.ylabel('NMSE')
    plt.title('NMSE vs SNR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # BER plot
    plt.figure(figsize=(6,4))
    plt.plot(snr_db, pnp_ber, marker='o', label='PnP sampling')
    plt.plot(snr_db, ls_ber, marker='s', label='Least Squares')
    plt.xlabel('SNR [dB]')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Phase Shift plot
    plt.figure(figsize=(6,4))
    plt.plot(snr_db, pnp_ps, marker='o', label='PnP sampling')
    plt.plot(snr_db, ls_ps, marker='s', label='Least Squares')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Phase Shift [deg]')
    plt.title('Phase Shift vs SNR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    my_plots()