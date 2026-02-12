import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from MixerLayer import Model
from myDiffuser import MyDDIMScheduler
from sp_modules import OFDMSignalGeneration
from config import get_config_pnp 
from utils import load_dataset

def compute_ber(x_true, y_true, H_est_, signal_generator):
	"""
	Compute BER using the original repo's decoder and error logic.
	x_true: (num_car,)
	y_true: (num_ant, num_car)
	H_est_: (num_ant, num_car)
	signal_generator: OFDMSignalGeneration instance
	"""
	try:
		# Zero-forcing equalization
		H_est_pinv = torch.linalg.pinv(H_est_)
		x_hat = torch.matmul(H_est_pinv, y_true)
		# Use the decoder from the signal generator (e.g., QPSK_decoder_hard)
		x_hat_demod = signal_generator.decode(x_hat)
		x_true_demod = signal_generator.decode(x_true)
		# Use the error calculation from SignalProcessingParams (utils.py)
		# Here, we use a tolerance-based error (real and imag parts within 0.01)
		err = x_hat_demod - x_true_demod
		ber = (err.real.abs().lt(0.01) * err.imag.abs().lt(0.01)).logical_not().float().mean().item()
	except Exception as e:
		print(f"BER computation error: {e}")
		ber = float('nan')
	return ber

@dataclass
class PnPResult:
	H_est: torch.Tensor
	H_final: torch.Tensor
	sigma_hat_H: torch.Tensor
	t0: int
	nmse: torch.Tensor
	ber: float


class PnPSampler:
	"""
	Plug-and-Play (PnP) posterior sampling wrapper that reuses the
	trained diffusion model, scheduler, and OFDM signal generation
	from the original repository.
	"""

	def __init__(
		self,
		config_path: str = "PnP_posterior_sampling/configs",
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

		self.signal_generator = OFDMSignalGeneration(self.config, self.config_noise, self.device)
		self.num_pilot = len(self.config.pilot_cars)
		self.num_car = self.config.num_car
		self.num_ant = self.config.num_ant

	def simulate_received(self, clean_channel: torch.Tensor):
		"""
		Simulate received signal Y using the original signal generator.
		Returns sp_params and the pilot-based initial channel estimate.
		"""
		sp_params = self.signal_generator.generate_signal(clean_channel)
		ori_channel = self.signal_generator.get_ori_channel()
		return sp_params, ori_channel

	def _compute_t0(self, sp_params) -> int:
		total_ener = torch.mean(1 + sp_params.noise_power_real / sp_params.sigma_H)
		target_alpha = (self.num_pilot / self.num_car) / total_ener
		return int(self.scheduler.get_time_by_alpha(target_alpha))

	def _estimate_sigma_hat(self, sp_params) -> torch.Tensor:
		sigma_hat = sp_params.sigma_H_sqrt
		return sigma_hat

	def _initialize_channel(
		self,
		ori_channel: torch.Tensor,
		sp_params,
		gamma: float,
		num_samples: int,
		num_remaining_channels: int,
	) -> Tuple[torch.Tensor, torch.Tensor, float]:
		total_ener = torch.mean(1 + sp_params.noise_power_real / sp_params.sigma_H)
		delta = math.sqrt(1 - gamma * gamma)

		channel_shape = ori_channel.shape
		bs = channel_shape[0]

		shape_sampled = torch.Size([bs, num_remaining_channels * num_samples]) + channel_shape[2:]
		channel_gen = torch.randn(shape_sampled, device=ori_channel.device, dtype=ori_channel.dtype)
		channel_gen_mask = torch.ones_like(channel_gen) - self.signal_generator.pilot_mask.unsqueeze(1)

		ori_est = torch.sqrt((self.num_car / self.num_pilot) / total_ener) * self.signal_generator.pilot_mask.unsqueeze(
			1
		) * ori_channel
		channel_estimated = gamma * ori_est

		channel = channel_estimated + delta * channel_gen * channel_gen_mask
		return channel, ori_est, delta

	def _normalize_channel(self, channel: torch.Tensor) -> torch.Tensor:
		power = channel.real * channel.real + channel.imag * channel.imag
		return channel / torch.sqrt(power.mean(dim=(-2, -1), keepdim=True))

	def _delta_t(self, step_idx: int, num_steps: int, schedule: str, alpha_bar_t: float, power: float) -> float:
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

	def run_pnp(
		self,
		clean_channel: torch.Tensor,
		lambda_reg: float,
		num_gen_steps: Optional[int] = None,
		gamma: Optional[float] = None,
		delta_t_schedule: str = "linear",
		delta_t_power: float = 1.0,
	) -> PnPResult:
		"""
		Run PnP posterior sampling using the trained diffusion model and
		the algorithm in my_algorithm.txt.
		Computes NMSE and BER as evaluation metrics.
		"""
		if num_gen_steps is None:
			num_gen_steps = self.config_diff.all_time_steps
		if gamma is None:
			gamma = self.config_diff.gamma

		clean_channel = clean_channel.to(self.device)
		sp_params, ori_channel = self.simulate_received(clean_channel)

		sigma_hat_H = self._estimate_sigma_hat(sp_params).view(-1, 1, 1, 1)
		t0 = self._compute_t0(sp_params)

		channel, ori_est, _ = self._initialize_channel(
			ori_channel, sp_params, gamma, self.config_diff.num_samples, self.config_diff.num_remaining_channels
		)

		step_size = max(1, int(round(t0 / num_gen_steps)))
		t = t0

		self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(channel.device)

		step_idx = 0
		with torch.no_grad():
			step_iter = tqdm(total=num_gen_steps, desc="PnP steps", leave=False)
			while t > 0:
				t_next = max(t - step_size, 0)

				shape = channel.shape
				shape_sampled = torch.Size([shape[0], -1]) + shape[2:]
				model_in = torch.flatten(channel, start_dim=0, end_dim=1)
				t_batch = torch.full(
					(model_in.shape[0],),
					t,
					device=channel.device,
					dtype=torch.int64,
				)
				pred = self.model(model_in, t_batch).reshape(shape_sampled)

				x = sp_params.X.expand(shape[0], shape[1], -1, -1)
				y = sp_params.Y.expand(shape[0], shape[1], -1, -1)

				alpha_t = self.scheduler.alphas[t]
				alpha_bar_t = self.scheduler.alphas_cumprod[t]
				alpha_bar_next = self.scheduler.alphas_cumprod[t_next]

				delta_t = self._delta_t(step_idx, num_gen_steps, delta_t_schedule, alpha_bar_t, delta_t_power)
				omega_t = delta_t
				rho_t = 1.0 - (lambda_reg * delta_t) / (1.0 + lambda_reg)

				H0t = pred
				#debug
				#H0t_prime = H0t + rho_t * ((1/sigma_hat_H) * y - H0t * x) * x.conj()
				H0t_prime = H0t + rho_t * ( y - H0t * x) * x.conj()

				eps_tilde = (channel - torch.sqrt(alpha_bar_t) * H0t_prime) / torch.sqrt(1 - alpha_bar_t)
				channel = torch.sqrt(alpha_bar_next) * H0t_prime + torch.sqrt(1 - alpha_bar_next) * omega_t * eps_tilde
				channel = self._normalize_channel(channel)

				t = t_next
				step_idx += 1
				step_iter.update(1)
			step_iter.close()

		H_final = channel
		H_est = sigma_hat_H * H_final

        # evaluation
		if H_est.dim() == 4 and clean_channel.dim() == 3:
			h_est_eval = H_est[:, 0]
		else:
			h_est_eval = H_est
		# NMSE: ||H_est - H||^2 / ||H||^2
		err = h_est_eval - clean_channel
		nmse = ((err.real ** 2 + err.imag ** 2).sum(dim=(-2, -1))) / ((clean_channel.real ** 2 + clean_channel.imag ** 2).sum(dim=(-2, -1)) + 1e-12)

		# BER: Use compute_ber utility
		x_true = sp_params.X[0, 0]  # shape: (num_car,)
		y_true = sp_params.Y[0, 0]  # shape: (num_ant, num_car)
		H_est_ = h_est_eval[0]      # shape: (num_ant, num_car)
		ber = compute_ber(x_true, y_true, H_est_, self.signal_generator)

		return PnPResult(H_est=H_est, H_final=H_final, sigma_hat_H=sigma_hat_H, t0=t0, nmse=nmse, ber=ber)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run PnP sampling on one batch")
	parser.add_argument("--config_path", type=str, default="PnP_posterior_sampling/configs")
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

	result = sampler.run_pnp(
		clean_channel,
		lambda_reg=args.lambda_reg,
		num_gen_steps=args.num_gen_steps,
		gamma=args.gamma,
		delta_t_schedule=args.delta_t_schedule,
		delta_t_power=args.delta_t_power,
	)
	print(f"t0={result.t0}")
	print(f"NMSE_mean={result.nmse.mean().item():.6f}")
	print(f"BER={result.ber:.6f}")


if __name__ == "__main__":
	main()
