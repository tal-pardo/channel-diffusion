"""
Clean DDIM-based channel sampling framework for diffusion receivers.
Decoupled from signal processing - generates channel realizations only.

Uses the same power normalization as original inference:
At each DDIM step, channels are normalized by their Frobenius norm to keep
values bounded during the denoising process.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm

from MixerLayer import Model
from myDiffuser import MyDDIMScheduler
from config import get_config


class DDIMChannelSampler:
    """
    DDIM sampling interface for trained diffusion channel model.
    
    Generates channel matrix realizations in frequency domain.
    Uses power normalization at each DDIM step (divide by Frobenius norm)
    to match the original inference process.
    
    Output channels are power-normalized but NOT mean-std denormalized,
    matching the scale of the normalized training data.
    """
    
    def __init__(self, 
                 config_path: str = 'results',
                 checkpoint_path: str = 'results/model_epoch50.pth',
                 device: str = 'cuda'):
        """
        Initialize sampler with trained model and configs.
        
        Args:
            config_path: Path to experiment directory with configs (default: 'results')
            checkpoint_path: Path to model checkpoint (.pth file) (default: 'results/model_epoch50.pth')
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load configurations from config_path
        self.config, self.config_car, self.config_ant, self.config_diff, self.config_noise = \
            get_config(config_path)
        self.num_ant = self.config.num_ant
        self.num_car = self.config.num_car
        
        print(f"✓ Loaded configs from: {config_path}")
        print(f"  → num_ant={self.num_ant}, num_car={self.num_car}")
        
        # Initialize model
        self.model = Model(self.config, self.config_car, self.config_ant)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize scheduler (same as training)
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
            rescale_betas_zero_snr=self.config_diff.rescale_betas_zero_snr
        )
        
    def sample(self, 
               received_signal: torch.Tensor,
               batch_size: int = 16,
               num_inference_steps: int = 50,
               progress_bar: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate channel samples from pure noise using DDIM.
        
        Requires received signal Y. Computes sigma_H = sqrt(MSE(Y)),
        normalizes Y by sigma_H, generates normalized channels,
        and denormalizes to recover true channel scale.
        
        Args:
            received_signal: (batch_size, num_ant, num_car) [complex64]
                            Received signal Y. Required for sigma_H computation.
            batch_size: Number of samples to generate
            num_inference_steps: Number of DDIM denoising steps (tradeoff: speed vs quality)
            progress_bar: Show progress bar
        
        Returns:
            (channels, Y_normalized):
                channels: (batch_size, num_ant, num_car) [complex64]
                         Denormalized channel estimates
                Y_normalized: (batch_size, num_ant, num_car) [complex64]
                             Normalized received signal Y/sigma_H_sqrt
        """
        # Compute sigma_H from received signal: sigma_H = sqrt(MSE(Y))
        sigma_H = torch.mean(
            received_signal.real * received_signal.real + 
            received_signal.imag * received_signal.imag,
            dim=(1, 2), keepdim=True
        )
        sigma_H_sqrt = torch.sqrt(sigma_H)
        
        # Normalize received signal
        Y_normalized = received_signal / sigma_H_sqrt
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Start with pure noise
        channels = torch.randn(
            batch_size, self.num_ant, self.num_car,
            dtype=torch.complex64,
            device=self.device
        )
        
        # Iterative denoising - matches original inference
        # Create iterator over timestep pairs (t_i, t_i-1)
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:])) + [(timesteps[-1], torch.tensor(-1, device=self.device))]
        iterator = tqdm(timestep_pairs, disable=not progress_bar, desc="DDIM Sampling")
        
        with torch.no_grad():
            for t_current, t_next in iterator:
                # Expand timesteps for batch
                t_batch = torch.full((batch_size,), t_current.item(), 
                                    dtype=torch.int64, device=self.device)
                
                # Model prediction
                pred = self.model(channels, t_batch)
                
                # Scheduler step - compute next denoised sample
                # Pass both current and next timesteps for DDIM transition formula
                step_output = self.scheduler.step(
                    model_output=pred,
                    timestep=t_current.item(),
                    prev_timestep=t_next.item(),
                    sample=channels,
                    num_samples=1
                )
                channels = step_output.prev_sample
                
                # Power normalization - same as in original generate_step()
                # Normalize by Frobenius norm to keep values bounded
                channel_power = channels.real * channels.real + channels.imag * channels.imag
                channels = channels / torch.sqrt(channel_power.mean(dim=(1, 2), keepdim=True))
        
        # Denormalize channels by sigma_H
        channels = channels * sigma_H_sqrt
        
        return channels, Y_normalized
    
    def sample_from_init(self,
                        init_channel: torch.Tensor,
                        received_signal: torch.Tensor,
                        num_inference_steps: int = 50,
                        init_noise_level: float = 0.5,
                        progress_bar: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples starting from an initial LS estimate using DDIM.
        
        Useful for refinement of pilot-based channel estimates.
        Requires received signal Y for sigma_H computation.
        
        Args:
            init_channel: (batch_size, num_ant, num_car) [complex64]
                         Initial channel estimate from LS (e.g., from pilots)
            received_signal: (batch_size, num_ant, num_car) [complex64]
                           Received signal Y. Required for sigma_H computation.
            num_inference_steps: DDIM steps
            init_noise_level: How much noise to add to initialization (0-1)
                             0 = pure init, 1 = pure noise
            progress_bar: Show progress
        
        Returns:
            (channels, Y_normalized):
                channels: (batch_size, num_ant, num_car) [complex64]
                         Refined denormalized channels
                Y_normalized: (batch_size, num_ant, num_car) [complex64]
                             Normalized received signal Y/sigma_H_sqrt
        """
        batch_size = init_channel.shape[0]
        
        # Compute sigma_H from received signal: sigma_H = sqrt(MSE(Y))
        sigma_H = torch.mean(
            received_signal.real * received_signal.real + 
            received_signal.imag * received_signal.imag,
            dim=(1, 2), keepdim=True
        )
        sigma_H_sqrt = torch.sqrt(sigma_H)
        
        # Normalize received signal
        Y_normalized = received_signal / sigma_H_sqrt
        
        # Normalize initialization to model space
        init_channel_normalized = init_channel / sigma_H_sqrt
        
        # Add noise to initialization
        noise = torch.randn_like(init_channel_normalized)
        alpha = 1 - init_noise_level
        beta = init_noise_level
        
        channels = alpha * init_channel_normalized + beta * noise
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Find appropriate starting timestep based on noise level
        start_step = int(init_noise_level * len(timesteps))
        timesteps = timesteps[start_step:]
        
        # Create timestep pairs for DDIM transitions
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:])) + [(timesteps[-1], torch.tensor(-1, device=self.device))]
        iterator = tqdm(timestep_pairs, disable=not progress_bar, desc="DDIM Refinement")
        
        with torch.no_grad():
            for t_current, t_next in iterator:
                t_batch = torch.full((batch_size,), t_current.item(),
                                    dtype=torch.int64, device=self.device)
                
                pred = self.model(channels, t_batch)
                step_output = self.scheduler.step(
                    model_output=pred,
                    timestep=t_current.item(),
                    prev_timestep=t_next.item(),
                    sample=channels,
                    num_samples=1
                )
                channels = step_output.prev_sample
                
                # Power normalization - same as original inference
                channel_power = channels.real * channels.real + channels.imag * channels.imag
                channels = channels / torch.sqrt(channel_power.mean(dim=(1, 2), keepdim=True))
        
        # Denormalize channels by sigma_H
        channels = channels * sigma_H_sqrt
        
        return channels, Y_normalized
    
    def sample_batch(self,
                     received_signal: torch.Tensor,
                     batch_size: int = 32,
                     num_inference_steps: int = 50,
                     init_channel: Optional[torch.Tensor] = None,
                     init_noise_level: float = 0.5,
                     save_path: Optional[str] = None,
                     progress_bar: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate many samples in batches, either from noise or initial estimates.
        
        Args:
            received_signal: (num_samples, num_ant, num_car) [complex64]
                           Received signal Y. Required for sigma_H computation.
            batch_size: Batch size for generation
            num_inference_steps: DDIM steps per batch
            init_channel: (num_samples, num_ant, num_car) [complex64] optional
                         Initial channel estimates for refinement.
                         If None, samples from pure noise.
            init_noise_level: Noise level for initialization (0-1, only used if init_channel provided)
                             0 = pure init, 1 = pure noise
            save_path: Optional path to save channels as .pt file
            progress_bar: Show progress bars during generation
        
        Returns:
            (all_channels, all_Y_normalized):
                all_channels: (num_samples, num_ant, num_car) [complex64]
                             Denormalized channel samples
                all_Y_normalized: (num_samples, num_ant, num_car) [complex64]
                                 Normalized received signals
        """
        num_samples = received_signal.shape[0]
        all_channels = []
        all_Y_normalized = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Handle last batch potentially being smaller
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # Extract batch-specific received signal
            batch_received_signal = received_signal[batch_start:batch_end]
            
            # Choose sampling method based on whether init_channel is provided
            if init_channel is not None:
                batch_init = init_channel[batch_start:batch_end]
                channels, Y_norm = self.sample_from_init(
                    init_channel=batch_init,
                    received_signal=batch_received_signal,
                    num_inference_steps=num_inference_steps,
                    init_noise_level=init_noise_level,
                    progress_bar=True
                )
            else:
                channels, Y_norm = self.sample(
                    received_signal=batch_received_signal,
                    batch_size=current_batch_size,
                    num_inference_steps=num_inference_steps,
                    progress_bar=True
                )
            
            all_channels.append(channels.cpu())
            all_Y_normalized.append(Y_norm.cpu())
        
        all_channels = torch.cat(all_channels, dim=0)
        all_Y_normalized = torch.cat(all_Y_normalized, dim=0)
        
        if save_path:
            torch.save({
                'channels': all_channels,
                'Y_normalized': all_Y_normalized
            }, save_path)
            print(f"Saved {num_samples} samples to {save_path}")
        
        return all_channels, all_Y_normalized
    
    def sample_with_condition(self,
                             condition_fn: Callable[[torch.Tensor], torch.Tensor],
                             received_signal: torch.Tensor,
                             batch_size: int = 16,
                             num_inference_steps: int = 50,
                             guidance_scale: float = 1.0,
                             progress_bar: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples with optional guidance function and power normalization.
        
        The guidance function is called at each step to enforce constraints.
        Requires received signal Y for sigma_H computation.
        
        Args:
            condition_fn: Function that takes (channels) and returns modified channels
                         Example: Force channel at pilot positions to known values
            received_signal: (batch_size, num_ant, num_car) [complex64]
                           Received signal Y. Required for sigma_H computation.
            batch_size: Number of samples
            num_inference_steps: DDIM steps
            guidance_scale: How much to apply guidance (1.0 = no guidance)
            progress_bar: Show progress
        
        Returns:
            (channels, Y_normalized):
                channels: (batch_size, num_ant, num_car) [complex64]
                         Denormalized channel samples with guidance
                Y_normalized: (batch_size, num_ant, num_car) [complex64]
                             Normalized received signal
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Compute sigma_H from received signal
        sigma_H = torch.mean(
            received_signal.real * received_signal.real + 
            received_signal.imag * received_signal.imag,
            dim=(1, 2), keepdim=True
        )
        sigma_H_sqrt = torch.sqrt(sigma_H)
        
        # Normalize received signal
        Y_normalized = received_signal / sigma_H_sqrt
        
        # Start with noise
        channels = torch.randn(batch_size, self.num_ant, self.num_car,
                              dtype=torch.complex64, device=self.device)
        
        # Create timestep pairs for DDIM transitions
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:])) + [(timesteps[-1], torch.tensor(-1, device=self.device))]
        iterator = tqdm(timestep_pairs, disable=not progress_bar, desc="Guided DDIM Sampling")
        
        with torch.no_grad():
            for t_current, t_next in iterator:
                t_batch = torch.full((batch_size,), t_current.item(),
                                    dtype=torch.int64, device=self.device)
                
                # Model prediction
                pred = self.model(channels, t_batch)
                
                # Apply guidance (condition enforcement)
                if guidance_scale != 1.0:
                    pred_conditioned = condition_fn(pred)
                    pred = (1 - guidance_scale) * pred + guidance_scale * pred_conditioned
                
                # Scheduler step
                step_output = self.scheduler.step(
                    model_output=pred,
                    timestep=t_current.item(),
                    prev_timestep=t_next.item(),
                    sample=channels,
                    num_samples=1
                )
                channels = step_output.prev_sample
                
                # Power normalization - same as original inference
                channel_power = channels.real * channels.real + channels.imag * channels.imag
                channels = channels / torch.sqrt(channel_power.mean(dim=(1, 2), keepdim=True))
        
        # Denormalization: scale by received signal magnitude
        channels = channels * sigma_H_sqrt
        
        return channels, Y_normalized


def main_example():
    """Example usage of the channel sampler."""
    
    # Initialize sampler with configs from results/
    sampler = DDIMChannelSampler(
        config_path='results',
        checkpoint_path='results/model_epoch50.pth',
        device='cuda'
    )
    
    print("=" * 60)
    print("DDIM Channel Sampler - Examples with Required Y")
    print("=" * 60)
    
    # Create synthetic received signals for demonstration
    batch_size_demo = 4
    num_samples_demo = 16
    num_ant = sampler.num_ant
    num_car = sampler.num_car
    device = sampler.device
    
    # Example 1: Sampling from pure noise
    print("\n1. Sampling from Pure Noise (requires Y)")
    print("-" * 60)
    
    snr_db = 10
    snr_linear = 10 ** (snr_db / 10)
    signal_power = 1.0 / snr_linear
    
    Y_demo = torch.randn(batch_size_demo, num_ant, num_car, 
                         dtype=torch.complex64, device=device) * torch.sqrt(torch.tensor(signal_power))
    
    print(f"Created received signal Y: shape={Y_demo.shape}")
    sigma_H_actual = torch.mean(Y_demo.real**2 + Y_demo.imag**2)
    print(f"  MSE(Y) = {sigma_H_actual:.6f} (SNR = {snr_db} dB)")
    
    channels_from_noise, Y_norm = sampler.sample(
        received_signal=Y_demo,
        batch_size=batch_size_demo,
        num_inference_steps=50,
        progress_bar=True
    )
    
    print(f"\nGenerated channels shape: {channels_from_noise.shape}")
    print(f"Normalized Y shape: {Y_norm.shape}")
    
    sigma_H_channels = torch.mean(channels_from_noise.real**2 + channels_from_noise.imag**2, dim=(1,2))
    print(f"Channel power (MSE per sample):")
    print(f"  Mean: {sigma_H_channels.mean():.6f}")
    print(f"  Should be close to sigma_H: {sigma_H_actual:.6f}")
    
    # Example 2: Fast sampling
    print("\n2. Fast Sampling (20 steps - lower quality)")
    print("-" * 60)
    channels_fast, _ = sampler.sample(
        received_signal=Y_demo,
        batch_size=batch_size_demo,
        num_inference_steps=20,
        progress_bar=True
    )
    print(f"Fast generation complete: {channels_fast.shape}")
    sigma_H_fast = torch.mean(channels_fast.real**2 + channels_fast.imag**2, dim=(1,2))
    print(f"Channel power - Mean: {sigma_H_fast.mean():.6f}")
    
    # Example 3: Refinement from LS estimate
    print("\n3. Refinement from Initial LS Estimate (requires Y)")
    print("-" * 60)
    
    init_estimate = torch.randn(batch_size_demo, num_ant, num_car, 
                                dtype=torch.complex64, device=device) * 0.5
    
    print(f"Initial LS estimate shape: {init_estimate.shape}")
    print(f"  Power before refinement: {torch.mean(init_estimate.real**2 + init_estimate.imag**2):.6f}")
    
    refined_channels, Y_norm_init = sampler.sample_from_init(
        init_channel=init_estimate,
        received_signal=Y_demo,
        num_inference_steps=50,
        init_noise_level=0.3,
        progress_bar=True
    )
    
    print(f"\nRefined channels shape: {refined_channels.shape}")
    sigma_H_refined = torch.mean(refined_channels.real**2 + refined_channels.imag**2, dim=(1,2))
    print(f"Channel power after refinement:")
    print(f"  Mean: {sigma_H_refined.mean():.6f}")
    print(f"  Should match sigma_H: {sigma_H_actual:.6f}")
    
    # Example 4: Batch sampling from noise
    print("\n4. Batch Generation from Pure Noise")
    print("-" * 60)
    
    Y_batch = torch.randn(num_samples_demo, num_ant, num_car,
                         dtype=torch.complex64, device=device) * torch.sqrt(torch.tensor(signal_power))
    
    print(f"Created batch of {num_samples_demo} received signals")
    
    channels_batch, Y_norm_batch = sampler.sample_batch(
        received_signal=Y_batch,
        batch_size=4,
        num_inference_steps=50,
        init_channel=None,
        save_path='generated_channels_noise.pt'
    )
    
    print(f"\nGenerated batch shape: {channels_batch.shape}")
    print(f"Normalized Y batch shape: {Y_norm_batch.shape}")
    
    sigma_H_batch = torch.mean(channels_batch.real**2 + channels_batch.imag**2, dim=(1,2))
    print(f"Channel power statistics (MSE per sample):")
    print(f"  Mean: {sigma_H_batch.mean():.6f}")
    print(f"  Std: {sigma_H_batch.std():.6f}")
    print(f"  Range: [{sigma_H_batch.min():.6f}, {sigma_H_batch.max():.6f}]")
    
    # Example 5: Batch sampling with refinement from LS
    print("\n5. Batch Generation with LS Refinement")
    print("-" * 60)
    
    init_batch = torch.randn(num_samples_demo, num_ant, num_car,
                            dtype=torch.complex64, device=device) * 0.5
    
    print(f"Created batch of {num_samples_demo} initial LS estimates")
    
    channels_batch_refined, Y_norm_batch_refined = sampler.sample_batch(
        received_signal=Y_batch,
        batch_size=4,
        num_inference_steps=50,
        init_channel=init_batch,
        init_noise_level=0.3,
        save_path='generated_channels_refined.pt'
    )
    
    print(f"\nRefined batch shape: {channels_batch_refined.shape}")
    
    sigma_H_batch_refined = torch.mean(channels_batch_refined.real**2 + channels_batch_refined.imag**2, dim=(1,2))
    print(f"Channel power after refinement:")
    print(f"  Mean: {sigma_H_batch_refined.mean():.6f}")
    print(f"  Std: {sigma_H_batch_refined.std():.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("✓ All sampling methods require received_signal Y")
    print("✓ sigma_H computed from Y: MSE(|Y|²)")
    print("✓ Y is normalized: Y_norm = Y / sqrt(sigma_H)")
    print("✓ Channels are denormalized: H = H_norm * sqrt(sigma_H)")
    print("✓ 3 sampling modes: from_noise, from_init (LS), batch (either mode)")
    print("=" * 60)


if __name__ == "__main__":
    main_example()
