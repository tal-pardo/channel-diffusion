import argparse
from email import parser
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
from tqdm import tqdm

from MixerLayer import Model				    
from myDiffuser import MyDDIMScheduler			
from pnp_posterior_sampling.pnp_utils import get_config_pnp , load_dataset		


@dataclass
class PnpResult:
	H_est: torch.Tensor
	gamma: float
	t0: int
	nmse: torch.Tensor
	ber: float
	phase_error_H: float
	phase_error_Y: float

@dataclass
class PnP_sp_params:
    Y: torch.Tensor				# shape: (bs, num_ant, num_car)
    x: torch.Tensor				# shape: (bs, num_car)
    sigma_H_hat: torch.Tensor	# shape: (bs,)

class PnP_SignalGenerator:
	"""
	Wrapper to all signal processing functions to be used in PnP sampling.
	Provides methods to simulate received signal, reconstruct symbols, and compute BER.
	"""

	def __init__(self, config, noise_config, device):
		self.num_ant = config.num_ant
		self.num_car = config.num_car
		self.sigma_n = noise_config.noise_power
		self.device = device
		self.config = config

		# Generate pilot mask and data mask 
		self.pilot_mask = torch.zeros(self.num_car, device=self.device)
		self.pilot_mask[config.pilot_cars] = 1.0						# shape: (num_car,)
		self.data_mask = 1.0 - self.pilot_mask							# shape: (num_car,)
		self.pilots = (1.0+0.0j) * self.pilot_mask						# shape: (num_car,)
	
	def generate_QPSK_x(self, bs):
		'''generate normalized QPSK symbols of shape (bs, num_car) 
		with random QPSK in data indices and ones in pilot indices'''
		shape = (bs, self.num_car)
		xx_r = (torch.randint(2, shape, device=self.device) - 0.5) * 1.414213562
		xx_i = (torch.randint(2, shape, device=self.device) - 0.5) * 1.414213562
		xx = xx_r * self.data_mask + 1j * xx_i * self.data_mask + self.pilots
		return xx.to(dtype=torch.complex64, device=self.device)
		
	def simulate_received(self, clean_channel: torch.Tensor):
		"""
		Simulate received signal Y=H@X+N using the PnP signal generator.
		Returns: X, Y, signa_H
		"""
		x = self.generate_QPSK_x(clean_channel.shape[0])   								# shape: (bs, num_car)
		diag_x = torch.diag_embed(x)    						     					# shape: (bs, num_car, num_car) 						   
		N_ = torch.randn_like(clean_channel) + 1j * torch.randn_like(clean_channel) 
		N = N_ * self.sigma_n        													# shape: (bs, num_ant, num_car)
		Y = torch.matmul(clean_channel, diag_x) + N    									# shape: (bs, num_ant, num_car)
		# sigma_hat_H = ||Y|| / sqrt(num_ant * num_car) as a rough estimate of channel power
		self.sigma_H_hat = torch.norm(Y, dim=(1,2)) / math.sqrt(self.num_ant * self.num_car)	# shape: (bs,)
		pnp_sp_params = PnP_sp_params(Y=Y.to(self.device), x=x.to(self.device), sigma_H_hat=self.sigma_H_hat.to(self.device))
		return pnp_sp_params
	
	def reconstruct_x(self, Y, H_est) -> torch.Tensor:
		# reconstruct x_hat using Maximum Ratio Combining (MRC)

		# Numerator: Sum( y * h_conj ) over antennas (dim=-2)
		hy = (Y * H_est.conj()).mean(dim=-2, keepdim=True)
		
		# Denominator: Sum( |h|^2 ) over antennas (dim=-2)
		hh = (H_est * H_est.conj()).real.mean(dim=-2, keepdim=True)

		# divide by sigma_H to return to the normalized symbol domain
		view_shape = (-1, *[1 for _ in hy.shape[1:]])
		sigma_reshaped = self.sigma_H_hat.view(view_shape)
		x_hat = hy / (hh * sigma_reshaped)

		# ensure pilots are exactly reconstructed
		x_hat = x_hat.real * self.data_mask + 1j * (x_hat.imag * self.data_mask) + self.pilots

		return x_hat
	
	def decoder(self, x) -> torch.Tensor:
		'''decode signal x to normalized QPSK symbols'''
		norm = 1/math.sqrt(2)
		return torch.sign(x.real) * norm + 1j * torch.sign(x.imag) * norm

	
	def compute_BER_1(self, x_true, Y, H_est) -> float:
		"""
		Compute BER across the batch
		x_true: (bs,num_car)
		Y: (bs, num_ant, num_car)
		H_est: (bs, num_ant, num_car)
		"""
		
		# reconstruct with MRC
		x_hat = self.reconstruct_x(Y, H_est)
		# Decode (Hard Decision)
		x_hat_demod = self.decoder(x_hat)
		# ensure pilots are exactly reconstructed
		x_hat_fix_pilots = x_hat_demod * self.data_mask + self.pilots		

		# compute BER with tolerance 
		err = x_hat_fix_pilots.squeeze(1) - x_true
		is_correct = (err.real.abs().lt(0.01) & err.imag.abs().lt(0.01))
		ber = (~is_correct).float().mean().item()
		return ber
	
	def compute_BER_2(self, x_true, Y, H_est) -> float:
		"""
		Compute BER across the batch
		x_true: (bs,num_car)
		Y: (bs, num_ant, num_car)
		H_est: (bs, num_ant, num_car)
		"""
		
		# reconstruct with MRC
		x_hat = self.reconstruct_x(Y, H_est)
		# Decode (Hard Decision)
		x_hat_demod = self.decoder(x_hat.squeeze(1))		# shape: (bs, num_car)
		x_hat_fix_pilots = x_hat_demod * self.data_mask + self.pilots
		
		real_correct = (torch.sign(x_hat_fix_pilots.real) == torch.sign(x_true.real))
		imag_correct = (torch.sign(x_hat_fix_pilots.imag) == torch.sign(x_true.imag))
        
        # 3. Compute BER on Data Indices Only
        # (Pilots are forced to match, so we ignore them to be rigorous)
		data_bool = self.data_mask.bool()
        
        # Total wrong bits in the data payload
		err_bits = (~real_correct[:, data_bool]).sum() + (~imag_correct[:, data_bool]).sum()
		total_data_bits = data_bool.sum() * 2 * H_est.shape[0] # 2 bits per symbol
		ber = err_bits.float() / total_data_bits
		return ber.item()
	
	def compute_NMSE(self, H_est, H_true) -> torch.Tensor:
		'''compute mean NMSE across the batch'''
		err = H_est - H_true
		nmse = ((err.abs()**2).sum(dim=(-2, -1))) / ((H_true.abs()**2).sum(dim=(-2, -1)) + 1e-12)
		return nmse.mean()
	


class PnPSampler:
	"""
	Plug-and-Play (PnP) posterior sampling wrapper that reuses the
	trained diffusion model, scheduler, and OFDM signal generation
	from the original repository.
	"""

	def __init__(
		self,
		config_path: str = "pnp_posterior_sampling/configs",
		checkpoint_path: str = "results/model_epoch50.pth",
		device: Optional[str] = None,
	):
		# Use explicit _pnp config files
		config_main = f"{config_path}/config_pnp.json"
		config_car = f"{config_path}/config_car_pnp.json"
		config_ant = f"{config_path}/config_ant_pnp.json"
		config_diff = f"{config_path}/config_diff_pnp.json"
		config_noise = f"{config_path}/config_noise_pnp.json"

		self.config, self.config_car, self.config_ant, self.config_diff, self.config_noise = get_config_pnp(
			config_main, config_car, config_ant, config_diff, config_noise
		)

		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)

		self.model = Model(self.config, self.config_car, self.config_ant).to(self.device)
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		self.model.load_state_dict(checkpoint)
		self.model.eval()

		self.scheduler = MyDDIMScheduler(
			num_train_timesteps=self.config_diff.num_train_timesteps,
			beta_start=self.config_diff.beta_start,
			beta_end=self.config_diff.beta_end,
			beta_schedule=self.config_diff.beta_schedule,
			trained_betas=None,
			clip_sample=self.config_diff.clip_sample,
			set_alpha_to_one=self.config_diff.set_alpha_to_one,
			steps_offset=self.config_diff.steps_offset,
			prediction_type=self.config_diff.prediction_type,
			thresholding=self.config_diff.thresholding,
			dynamic_thresholding_ratio=self.config_diff.dynamic_thresholding_ratio,
			clip_sample_range=self.config_diff.clip_sample_range,
			sample_max_value=self.config_diff.sample_max_value,
			timestep_spacing=self.config_diff.timestep_spacing,
			rescale_betas_zero_snr=self.config_diff.rescale_betas_zero_snr,
		)

		self.pnp_signal_generator = PnP_SignalGenerator(self.config, self.config_noise, self.device)
		self.pilot_mask = self.pnp_signal_generator.pilot_mask  		# shape: (num_car,) {0,1}
		self.data_mask = 1.0 - self.pilot_mask							# shape: (num_car,) {0,1}
		self.pilots = (1.0+0.0j) * self.pilot_mask		            	# shape: (num_car,) {complex tensor with 1 in pilot positions and 0 in data positions}	
		self.num_pilot = len(self.config.pilot_cars)
		self.num_car = self.config.num_car
		self.num_ant = self.config.num_ant
		self.sigma_n = self.config_noise.noise_power


	def compute_t0_gamma(self, target_alpha) -> int:
		# |P| / |P|+|D| = num_pilot / num_car
		ratio = self.num_pilot / self.num_car
		target_alpha = 0.8
		gamma = math.sqrt(target_alpha/ratio)
		
		# Find t0 as the first t where alphabar_t <= target_alpha
		self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
		err = target_alpha - self.scheduler.alphas_cumprod
		# t0 = argmin |alphabar_t - target_alpha|
		t0 = (err.abs()).argmin().item() 
		return t0 , gamma


	def initialize_channel(self, gamma: float, pnp_sp_params: dataclass) -> Tuple[torch.Tensor, torch.Tensor, float]:
		
		# delta = sqrt(1 - gamma^2 * (sigma_n^2/sigma_H^2) * (|P|/(|P|+|D|)))
		sigma_n2 = self.config_noise.noise_power ** 2			        # sigma_n^2: scalar
		sigma_H2 = pnp_sp_params.sigma_H_hat.mean() ** 2		        # sigma_H^2: scalar
		ratio = self.num_pilot / self.num_car
		try: 
			delta = math.sqrt(1 - pow(gamma, 2) * (sigma_n2 / sigma_H2) * ratio)
		except ValueError as e:
			print(f"Error: computed negative delta: {e}")
			raise e

		# initial channel
		# for i in Pilot positions: H0 = gamma * Y / (x * sigma_H_hat) + delta * noise
		# x in pilot positions is 1, so we can simplify to Y / sigma_H_hat
		H_pilot = pnp_sp_params.Y * self.pilot_mask / pnp_sp_params.sigma_H_hat.view(-1, 1, 1)		# shape: (bs, num_ant, num_car)
		# for i in Data positions: H0 = delta * noise
		noise = (torch.randn(H_pilot.shape, device=self.device) + 1j * torch.randn(H_pilot.shape, device=self.device)) / math.sqrt(2)
		H_data = noise * self.data_mask

		channel = gamma * H_pilot + delta * H_data
		
		return channel, H_pilot
	

	def normalize_channel(self, channel: torch.Tensor) -> torch.Tensor:
		'''norm by RMS each sample in batch 
		output: H / sqrt( 1/N * sum(|Hi,j|^2))'''
		power = channel.real * channel.real + channel.imag * channel.imag
		return channel / torch.sqrt(power.mean(dim=(-2, -1), keepdim=True))

	def delta_t(self, step_idx: int, num_steps: int, schedule: str, alpha_bar_t: float, power: float) -> float:
		if num_steps <= 1:
			return 0.0
			
		progress = step_idx / float(num_steps - 1)
		if schedule == "linear":
			return max(0.0, 1.0 - progress)
		if schedule == "cosine":
			return float(math.cos(0.5 * math.pi * progress))
		if schedule == "alphabar":
			return float(alpha_bar_t ** power)
		raise ValueError(f"Unsupported delta_t schedule: {schedule}")
	
	
	def compute_phase_error(self, H_est: torch.Tensor, clean_channel: torch.Tensor) -> float:
		# Compute the complex correlation between the estimated and true channels
		correlation = (clean_channel * H_est.conj()).sum(dim=(-2, -1))
		
		# Extract the phase difference
		phase_error = torch.angle(correlation).mean().item()  # Average phase error across the batch
		
		return math.degrees(phase_error)  # Convert to degrees for interpretability

	def run_pnp_v2(
		self,
		clean_channel: torch.Tensor,
		lambda_reg: float=0.1,
		num_gen_steps: Optional[int] = 100,
		target_alpha: Optional[float] = 0.9,
		xi: float = 1.0,
		delta_t_schedule: str = "linear",
		delta_t_power: float = 1.0,
	) -> PnpResult:
		"""
		Run PnP posterior sampling using the trained diffusion model and
		the algorithm in my_algorithm.txt.
		Computes NMSE and BER as evaluation metrics.
		"""
		if num_gen_steps is None:
			num_gen_steps = self.config_diff.all_time_steps
		'''if gamma is None:
			gamma = self.config_diff.gamma'''

		clean_channel = clean_channel.to(self.device)				 	# shape: (bs, num_ant, num_car)
		pnp_sp_params = self.pnp_signal_generator.simulate_received(clean_channel)
		sigma_hat_H = pnp_sp_params.sigma_H_hat.view(-1, 1, 1)		    # shape: (bs, 1, 1) for broadcasting
		Y = pnp_sp_params.Y
		t0 , gamma_ = self.compute_t0_gamma(target_alpha)

		channel, H_pilot = self.initialize_channel(gamma_, pnp_sp_params)
		channel = self.normalize_channel(channel)
		step_size = max(1, int(round(t0 / num_gen_steps)))
		t = t0

		self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(channel.device)

		step_idx = 0
		with torch.no_grad():
			#step_iter = tqdm(total=num_gen_steps, desc="PnP steps", leave=False)
			while t > 0:
				t_next = max(t - step_size, 0)
				t_batch = torch.full((channel.shape[0],), t, device=channel.device, dtype=torch.int64)
				pred = self.model(channel, t_batch)

				x_hat_un_norm = self.pnp_signal_generator.reconstruct_x(Y, pred)	# shape: (bs, 1, num_car)
				x_hat_ = self.pnp_signal_generator.decoder(x_hat_un_norm)
				x_hat = x_hat_ * self.data_mask + self.pilots		# ensure pilots are exactly reconstructed

				alpha_bar_t = self.scheduler.alphas_cumprod[t]
				alpha_bar_next = self.scheduler.alphas_cumprod[t_next]

				# is delta_t the same for all samples in the batch? Yes, since it only depends on t and the schedule, not on the channel itself.
				delta_t = self.delta_t(step_idx, num_gen_steps, delta_t_schedule, alpha_bar_t, delta_t_power)
				omega_t = delta_t
				rho_t = 1.0 - (lambda_reg * delta_t) / (1.0 + lambda_reg)
				rho_mask = rho_t * (1.0 + xi * self.pilot_mask)

				H0t = pred	
				H0t_prime = H0t + rho_mask * ((1/ sigma_hat_H) * Y - H0t * x_hat) * x_hat.conj() 

				eps_tilde = (channel - torch.sqrt(alpha_bar_t) * H0t_prime) / torch.sqrt(1 - alpha_bar_t)
				channel = torch.sqrt(alpha_bar_next) * H0t_prime + torch.sqrt(1 - alpha_bar_next) * omega_t * eps_tilde
				channel = self.normalize_channel(channel)

				t = t_next
				step_idx += 1
				#step_iter.update(1)
			#step_iter.close()

		H_final_norm = channel               # shape: (bs, 1, num_ant, num_car)
		H_est = sigma_hat_H * H_final_norm
		
		# NMSE: ||H_est - H||^2 / ||H||^2
		nmse = self.pnp_signal_generator.compute_NMSE(H_est, clean_channel)

		# mean BER:      
		x_true = pnp_sp_params.x		# shape: (bs, num_car)
		ber = self.pnp_signal_generator.compute_BER_2(x_true, Y, H_est)

		# phase error in degrees
		phase_error_H = self.compute_phase_error(H_est, clean_channel)
		Y_est = H_est * x_true.unsqueeze(1)		# shape: (bs, num_ant, num_car)
		phase_error_Y = self.compute_phase_error(Y_est, Y)

		return PnpResult(H_est=H_est, gamma=gamma_, t0=t0, nmse=nmse, ber=ber, phase_error_H=phase_error_H, phase_error_Y=phase_error_Y)


def main_single_batch() -> None:
	parser = argparse.ArgumentParser(description="Run pnp sampling version 2 on one batch")
	parser.add_argument("--config_path", type=str, default="pnp_posterior_sampling/configs")
	parser.add_argument("--checkpoint_path", type=str, default="results/model_epoch50.pth")
	parser.add_argument("--lambda_reg", type=float, default=0.1)
	parser.add_argument("--num_gen_steps", type=int, default=100)
	parser.add_argument("--gamma", type=float, default=None)
	parser.add_argument("--delta_t_schedule", type=str, default="linear")
	parser.add_argument("--delta_t_power", type=float, default=1.0)
	args = parser.parse_args()

	# Always use explicit _pnp config files
	sampler = PnPSampler(
		config_path=args.config_path,
		checkpoint_path=args.checkpoint_path,
	)

	datasets = load_dataset(sampler.config)
	if len(datasets) < 2:
		raise ValueError("Dataset must provide a test split")
	
	data_loader = torch.utils.data.DataLoader(
		datasets[1],
		batch_size=sampler.config.bs,
		shuffle=False,
		num_workers=sampler.config.num_workers,
		drop_last=sampler.config.drop_last,
	)

	batch = next(iter(data_loader))
	if sampler.config.dataset == "deepmimo":
		clean_channel = batch[0]
	else:
		raise ValueError("Unsupported dataset")

	result = sampler.run_pnp_v2(
		clean_channel,
		lambda_reg=args.lambda_reg,
		num_gen_steps=args.num_gen_steps,
		target_alpha=args.target_alpha,
		xi=args.xi,
		delta_t_schedule=args.delta_t_schedule,
		delta_t_power=args.delta_t_power,
	)
	print(f"t0={result.t0}")
	print(f"gamma={result.gamma:.4f}")
	print(f"NMSE_mean={result.nmse.mean().item():.6f}")
	print(f"BER={result.ber:.10f}")
	print(f"Phase error between H_est and H_true: {result.phase_error_H:.2f} degrees")
	print(f"Phase error between Y_est and Y_true: {result.phase_error_Y:.2f} degrees")

def main_test() -> None:
    parser = argparse.ArgumentParser(description="Run pnp sampling version 2 on whole test dataset")
    parser.add_argument("--config_path", type=str, default="pnp_posterior_sampling/configs")
    parser.add_argument("--checkpoint_path", type=str, default="results/model_epoch50.pth")
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--num_gen_steps", type=int, default=100)
    parser.add_argument("--target_alpha", type=float, default=0.9)
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--delta_t_schedule", type=str, default="linear")
    parser.add_argument("--delta_t_power", type=float)
    args = parser.parse_args()

    # Initialize Sampler
    sampler = PnPSampler(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    # Load Dataset
    datasets = load_dataset(sampler.config)
    if len(datasets) < 2:
        raise ValueError("Dataset must provide a test split")
    
    data_loader = torch.utils.data.DataLoader(
        datasets[1],
        batch_size=sampler.config.bs,
        shuffle=False,
        num_workers=sampler.config.num_workers,
        drop_last=sampler.config.drop_last,
    )

    print(f"Starting evaluation on {len(data_loader)} batches...")

    # --- Initialize Accumulators ---
    total_nmse = 0.0
    total_ber = 0.0
    total_phase_h = 0.0
    total_phase_y = 0.0
    num_batches = 0

    # --- Loop Over Entire Test Dataset ---
    # We use tqdm here to show a progress bar for the batches
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating Dataset")):
        
        # Extract clean channel based on dataset type
        if sampler.config.dataset == "deepmimo":
            clean_channel = batch[0]
        else:
            raise ValueError("Unsupported dataset")

        # Run PnP on current batch
        result = sampler.run_pnp_v2(
            clean_channel,
            lambda_reg=args.lambda_reg,
            num_gen_steps=args.num_gen_steps,
            target_alpha=args.target_alpha,
			xi = args.xi,
            delta_t_schedule=args.delta_t_schedule,
            delta_t_power=args.delta_t_power,
        )

        # Accumulate metrics
        # Note: result.nmse is likely a tensor, so we use .item() to get the float
        nmse_val = result.nmse.item() if isinstance(result.nmse, torch.Tensor) else result.nmse
        
        total_nmse += nmse_val
        total_ber += result.ber
        total_phase_h += result.phase_error_H
        total_phase_y += result.phase_error_Y
        num_batches += 1

    # --- Compute and Print Final Averages ---
    avg_nmse = total_nmse / num_batches
    avg_ber = total_ber / num_batches
    avg_phase_h = total_phase_h / num_batches
    avg_phase_y = total_phase_y / num_batches

    print("\n" + "="*50)
    print(f"FINAL RESULTS (Averaged over {num_batches} batches)")
    print("lambda_reg: {:.4f}, target_alpha: {:.2f}, xi: {:.2f}, delta_t_schedule: {}, delta_t_power: {}".format(
        args.lambda_reg, args.target_alpha, args.xi, args.delta_t_schedule, args.delta_t_power))
    print("="*50)
    print(f"NMSE:             {avg_nmse:.6f}")
    print(f"BER:              {avg_ber:.8f}")
    print(f"Phase Error (H):  {avg_phase_h:.4f} degrees")
    print(f"Phase Error (Y):  {avg_phase_y:.4f} degrees")
    print("="*50)

def main_LS_test():
	from pnp_posterior_sampling.pnp_utils import LeastSquaresSampler
	
	parser = argparse.ArgumentParser(description="Run LS baseline on the whole test set")
	parser.add_argument("--config_path", type=str, default="pnp_posterior_sampling/configs")
	parser.add_argument("--checkpoint_path", type=str, default="results/model_epoch50.pth")
	args = parser.parse_args()

    # Load config and dataset
	config_main = f"{args.config_path}/config_pnp.json"
	config_car = f"{args.config_path}/config_car_pnp.json"
	config_ant = f"{args.config_path}/config_ant_pnp.json"
	config_diff = f"{args.config_path}/config_diff_pnp.json"
	config_noise = f"{args.config_path}/config_noise_pnp.json"
	config, config_car, config_ant, config_diff, config_noise = get_config_pnp(
        config_main, config_car, config_ant, config_diff, config_noise
    )
	device = "cuda" if torch.cuda.is_available() else "cpu"
	ls_sampler = LeastSquaresSampler(config, config_noise, device)
	
	datasets = load_dataset(config)
	if len(datasets) < 2:
		raise ValueError("Dataset must provide a test split")
	
	data_loader = torch.utils.data.DataLoader(
        datasets[1],
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    # --- Initialize Accumulators ---
	total_nmse = 0.0
	total_ber = 0.0
	total_phase_h = 0.0
	total_phase_y = 0.0
	num_batches = 0
	
	for batch in tqdm(data_loader, desc="LS Test Batches"):
		if config.dataset == "deepmimo":
			clean_channel = batch[0].to(device)
		else:
			raise ValueError("Unsupported dataset")

        # Simulate received signal and get pilots
		pnp_signal_generator = ls_sampler.pnp_signal_generator
		pnp_sp_params = pnp_signal_generator.simulate_received(clean_channel)
		Y = pnp_sp_params.Y
		x_true = pnp_sp_params.x

        # LS estimate
		H_ls = ls_sampler.estimate(Y)
		
		num_batches += 1

		# NMSE
		nmse = pnp_signal_generator.compute_NMSE(H_ls, clean_channel)
		total_nmse += nmse.item() if isinstance(nmse, torch.Tensor) else nmse

		# BER (use your decoder and BER function)
		ber = pnp_signal_generator.compute_BER_2(x_true, Y, H_ls)
		total_ber += ber


		# Initialize Sampler for evaluation functions
		pnp_sampler = PnPSampler(config_path=args.config_path,
			checkpoint_path=args.checkpoint_path,)

		# Phase error H
		phase_error_H = pnp_sampler.compute_phase_error(H_ls, clean_channel)
		total_phase_h += phase_error_H
		
		# Phase error Y
		Y_est = H_ls * x_true.unsqueeze(1)	
		phase_error_Y = pnp_sampler.compute_phase_error(Y_est, Y)
		total_phase_y += phase_error_Y

    # --- Compute and Print Final Averages ---
	avg_nmse = total_nmse / num_batches
	avg_ber = total_ber / num_batches
	avg_phase_h = total_phase_h / num_batches
	avg_phase_y = total_phase_y / num_batches

	noise_dB = 10 * math.log10(1/(config_noise.noise_power ** 2))

	print("\n" + "="*50)
	print(f"FINAL RESULTS LS estimation (Averaged over {num_batches} batches)")
	print("="*50)
	print(f"Noise Power (dB): {noise_dB:.2f}")
	print(f"NMSE:             {avg_nmse:.6f}")
	print(f"BER:              {avg_ber:.8f}")
	print(f"Phase Error (H):  {avg_phase_h:.4f} degrees")
	print(f"Phase Error (Y):  {avg_phase_y:.4f} degrees")
	print("="*50)


if __name__ == "__main__":
    main_LS_test()