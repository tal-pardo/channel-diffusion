"""
Visualization script for testing DDIMChannelSampler.

Tests both sampling methods:
1. Batch sampling from pure noise
2. Batch sampling from initial LS estimates

Compares with real samples from the dataset.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import os

from channel_sampler import DDIMChannelSampler
from sp_modules import QPSK_generation
from utils import load_dataset
from config import get_config


def generate_transmitted_signal(shape, config, device='cuda'):
    """
    Generate transmitted QPSK signal using data and pilot masks.
    
    Returns:
        X: (batch, 1, num_car) transmitted signal
    """
    # Get mask from signal generator
    X = QPSK_generation(
        shape=(shape[0], 1, shape[-1]),
        mask=None,
        dtype=torch.cfloat,
        device=device
    )
    return X


def create_received_signal(channel, config, device='cuda', snr_db=10):
    """
    Create received signal Y = H * X + noise.
    
    Args:
        channel: (batch, num_ant, num_car)
        config: configuration dict
        snr_db: SNR in dB
    
    Returns:
        Y: (batch, num_ant, num_car) received signal
        X: (batch, 1, num_car) transmitted signal
    """
    batch_size = channel.shape[0]
    num_car = channel.shape[-1]
    
    # Generate transmitted signal
    X = QPSK_generation(
        shape=(batch_size, 1, num_car),
        mask=None,
        dtype=torch.cfloat,
        device=device
    )  # (batch, 1, num_car)
    
    # Transmit through channel: Y = H * X
    Y = channel * X  # (batch, num_ant, num_car) * (batch, 1, num_car) -> (batch, num_ant, num_car)
    
    # Add noise based on SNR
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(Y.real**2 + Y.imag**2)
    noise_power = signal_power / snr_linear
    
    noise = torch.randn_like(Y) * torch.sqrt(noise_power)
    Y = Y + noise
    
    return Y, X


def create_ls_estimate(channel, pilot_indices, noise_power=0.01, device='cuda'):
    """
    Create LS pilot-based channel estimate.
    
    Simulates: H_ls = (Y_pilots * X_pilots^H) / |X_pilots|^2
    
    Args:
        channel: (batch, num_ant, num_car) true channel
        pilot_indices: indices of pilot subcarriers
        noise_power: power of noise
    
    Returns:
        H_ls: (batch, num_ant, num_car) LS estimate (sparse with pilots only)
    """
    batch_size, num_ant, num_car = channel.shape
    
    # Create pilot signal
    X_pilots = torch.ones((batch_size, 1, len(pilot_indices)), dtype=torch.cfloat, device=device)
    
    # Get pilot channel
    channel_pilots = channel[:, :, pilot_indices]  # (batch, num_ant, num_pilots)
    
    # Transmit pilots through channel
    Y_pilots = channel_pilots * X_pilots  # (batch, num_ant, num_pilots)
    
    # Add pilot noise
    pilot_noise = torch.randn_like(Y_pilots) * torch.sqrt(torch.tensor(noise_power))
    Y_pilots = Y_pilots + pilot_noise
    
    # LS estimation at pilot positions
    H_ls_pilots = Y_pilots / X_pilots  # (batch, num_ant, num_pilots)
    
    # Create full-length estimate with zeros at non-pilot positions
    H_ls = torch.zeros_like(channel)
    H_ls[:, :, pilot_indices] = H_ls_pilots
    
    return H_ls


def load_real_samples(config, num_samples=16, device='cuda'):
    """
    Load real channel samples from dataset using config.
    
    Returns:
        channels: (num_samples, num_ant, num_car) real channel samples
    """
    # Load directly from data.npy in dataset folder
    import numpy as np
    data_path = os.path.join('dataset', 'data.npy')
    
    if os.path.exists(data_path):
        channel_data = np.load(data_path)  # N*1*ant*car complex
        channel_torch = torch.tensor(channel_data).squeeze(1)  # N*ant*car cfloat tensor
        channel_torch = channel_torch * 1e5  # Apply same normalization as in training
    else:
        # Fallback to load_dataset if data.npy doesn't exist
        datasets = load_dataset(config)
        test_dataset = datasets[1]
        channel_list = [test_dataset[i][0] for i in range(len(test_dataset))]
        channel_torch = torch.stack(channel_list)
    
    # Get samples
    channels = []
    num_available = min(num_samples, channel_torch.shape[0])
    for i in range(num_available):
        channel = channel_torch[i]
        if isinstance(channel, torch.Tensor):
            channels.append(channel)
        else:
            channels.append(torch.tensor(channel, dtype=torch.cfloat))
    
    channels = torch.stack(channels).to(device=device, dtype=torch.cfloat)
    return channels


def visualize_batch(batch, title, fig_idx=1, vmin=None, vmax=None):
    """
    Visualize a batch of channels in magnitude.
    
    Args:
        batch: (batch_size, num_ant, num_car)
        title: title for the figure
        fig_idx: figure index
    """
    batch_size, num_ant, num_car = batch.shape
    
    # Move to CPU and get magnitude
    batch_mag = torch.abs(batch).cpu().numpy()
    
    # Create grid of subplots
    K = min(batch_size, 16)  # Show max 16 samples
    grid_size = int(np.ceil(np.sqrt(K)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten() if K > 1 else [axes]
    
    # Compute global vmin/vmax for consistent scaling
    if vmin is None:
        vmin = batch_mag[:K].min()
    if vmax is None:
        vmax = batch_mag[:K].max()
    
    for idx in range(K):
        ax = axes[idx]
        
        # Visualize mean over antenna dimension
        channel_mag = batch_mag[idx].mean(axis=0)  # Average over antennas
        
        im = ax.imshow(
            channel_mag[np.newaxis, :],
            aspect='auto',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        ax.set_title(f'Sample {idx}')
        ax.set_ylabel('Antenna')
        ax.set_xlabel('Subcarrier')
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(K, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, label='Magnitude')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, (vmin, vmax)


def main(args):
    """Main visualization pipeline."""
    
    print("=" * 80)
    print("DDIM Channel Sampler - Visualization Test")
    print("=" * 80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    K = args.num_samples
    snr_db = args.snr_db
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    
    # Load configuration
    print(f"\n1. Loading Configuration from {config_path}...")
    config, config_car, config_ant, _, _ = get_config(config_path)
    num_ant = config.num_ant
    num_car = config.num_car
    print(f"   ✓ Config loaded: {num_ant} antennas, {num_car} subcarriers")
    
    # Initialize sampler
    print(f"\n2. Initializing Channel Sampler...")
    sampler = DDIMChannelSampler(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )
    print(f"   ✓ Sampler ready")
    
    # ========== Method 1: Create Y received signals from test data ==========
    print(f"\n3. Creating {K} Received Signals (Y) from Random Channels...")
    # Load real channels from dataset
    real_channels = load_real_samples(config, num_samples=K, device=device)
    print(f"   ✓ Loaded real channels: {real_channels.shape}")
    
    # Generate Y signals from real channels
    Y_signals = []
    for i in tqdm(range(K), desc="   Generating Y signals"):
        channel = real_channels[i:i+1]  # (1, num_ant, num_car)
        Y, _ = create_received_signal(channel, config, device=device, snr_db=snr_db)
        Y_signals.append(Y)
    
    Y_signals = torch.cat(Y_signals, dim=0)  # (K, num_ant, num_car)
    print(f"   ✓ Y signals shape: {Y_signals.shape}")
    print(f"   ✓ Y power (MSE): {torch.mean(Y_signals.real**2 + Y_signals.imag**2):.6f}")
    
    # ========== Method 2: Sample from Pure Noise ==========
    print(f"\n3. Batch Sampling from Pure Noise (using Y)...")
    channels_from_noise, Y_norm_noise = sampler.sample_batch(
        received_signal=Y_signals,
        batch_size=args.batch_size,
        num_inference_steps=args.inference_steps,
        init_channel=None,
        progress_bar=True
    )
    print(f"   ✓ Generated channels: {channels_from_noise.shape}")
    print(f"   ✓ Channel power (MSE): {torch.mean(channels_from_noise.real**2 + channels_from_noise.imag**2, dim=(1,2)).mean():.6f}")
    
    # ========== Method 3: Sample from LS Estimates ==========
    print(f"\n4. Creating LS Pilot Estimates...")
    pilot_indices = list(range(0, num_car, num_car // 16))  # ~16 pilot subcarriers
    pilot_indices = pilot_indices[:16]  # Use exactly 16 pilots
    print(f"   ✓ Pilot subcarriers: {len(pilot_indices)} at indices {pilot_indices[:5]}...")
    
    H_ls_batch = torch.cat([
        create_ls_estimate(real_channels[i:i+1], pilot_indices, device=device)
        for i in range(K)
    ], dim=0)
    print(f"   ✓ LS estimates shape: {H_ls_batch.shape}")
    print(f"   ✓ LS power (MSE): {torch.mean(H_ls_batch.real**2 + H_ls_batch.imag**2, dim=(1,2)).mean():.6f}")
    
    print(f"\n5. Batch Sampling from LS Estimates (using Y)...")
    channels_from_init, Y_norm_init = sampler.sample_batch(
        received_signal=Y_signals,
        batch_size=args.batch_size,
        num_inference_steps=args.inference_steps,
        init_channel=H_ls_batch,
        init_noise_level=args.init_noise_level,
        progress_bar=True
    )
    print(f"   ✓ Refined channels: {channels_from_init.shape}")
    print(f"   ✓ Channel power (MSE): {torch.mean(channels_from_init.real**2 + channels_from_init.imag**2, dim=(1,2)).mean():.6f}")
    
    # ========== Method 4: Get Real Samples from Dataset ==========
    print(f"\n6. Loading {K} Real Channel Samples from Dataset...")
    channels_real = real_channels
    print(f"   ✓ Real channels shape: {channels_real.shape}")
    print(f"   ✓ Channel power (MSE): {torch.mean(channels_real.real**2 + channels_real.imag**2, dim=(1,2)).mean():.6f}")
    
    # ========== Visualization ==========
    print(f"\n7. Creating Visualizations...")
    
    # Find consistent scaling across all batches
    all_data = torch.cat([channels_real, channels_from_noise, channels_from_init], dim=0)
    all_mag = torch.abs(all_data)
    global_vmin = all_mag.min().item()
    global_vmax = all_mag.max().item()
    
    # Create visualizations
    fig1, _ = visualize_batch(
        channels_real,
        f"Real Dataset Channels (K={K})",
        fig_idx=1,
        vmin=global_vmin,
        vmax=global_vmax
    )
    
    fig2, _ = visualize_batch(
        channels_from_noise,
        f"Generated from Pure Noise (K={K}, Y-normalized)",
        fig_idx=2,
        vmin=global_vmin,
        vmax=global_vmax
    )
    
    fig3, _ = visualize_batch(
        channels_from_init,
        f"Refined from LS Estimates (K={K}, Y-normalized, init_noise={args.init_noise_level})",
        fig_idx=3,
        vmin=global_vmin,
        vmax=global_vmax
    )
    
    print(f"   ✓ Visualization complete")
    
    # ========== Summary Statistics ==========
    print(f"\n{'='*80}")
    print("Summary Statistics (Magnitude)")
    print(f"{'='*80}")
    
    for name, channels in [
        ("Real Dataset", channels_real),
        ("From Noise", channels_from_noise),
        ("From LS Init", channels_from_init)
    ]:
        mag = torch.abs(channels)
        stats = {
            'Mean': mag.mean().item(),
            'Std': mag.std().item(),
            'Min': mag.min().item(),
            'Max': mag.max().item(),
        }
        print(f"\n{name}:")
        for key, val in stats.items():
            print(f"  {key:10s}: {val:.6f}")
    
    print(f"\n{'='*80}")
    print("Visualization saved as PNG files")
    print(f"{'='*80}\n")
    
    # Save figures
    fig1.savefig('real_channels.png', dpi=150, bbox_inches='tight')
    fig2.savefig('sampled_from_noise.png', dpi=150, bbox_inches='tight')
    fig3.savefig('sampled_from_ls.png', dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize DDIM Channel Sampler outputs')
    
    parser.add_argument('--config_path', type=str, default='results',
                        help='Path to config directory (default: results)')
    parser.add_argument('--checkpoint_path', type=str, default='results/model_epoch50.pth',
                        help='Path to model checkpoint (default: results/model_epoch50.pth)')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize per method')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--inference_steps', type=int, default=50,
                        help='Number of DDIM inference steps')
    parser.add_argument('--init_noise_level', type=float, default=0.3,
                        help='Noise level for LS refinement (0-1)')
    parser.add_argument('--snr_db', type=float, default=10,
                        help='SNR in dB for received signal generation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)
