"""
Testing the algorithm on a structured Gaussian channel with non-diagonal autocovariance.
h = vec(H) ~ CN(0, Σ) 
E[h_i, h_i] = sqrt(N_a*N_c) for all i (to match normalized matrices)
"""

import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_structured_gaussian_channel(N_a, N_c, rho_ant=0.7, rho_car=0.7, device='cuda'):
    """
    Generate H ~ CN(0, Σ) with non-diagonal autocovariance.
    
    Σ = (N_a*N_c) * (Σ_ant ⊗ Σ_car)
    
    where Σ_ant, Σ_car are Toeplitz correlation matrices (unit diagonal),
    so that diag(Σ) = [N_a*N_c, N_a*N_c, ..., N_a*N_c].
    
    This gives E[||h||²] = tr(Σ) = (N_a*N_c)² and E[||h||] = N_a*N_c.
    """
    # Create correlation matrices (Toeplitz, unit diagonal)
    sigma_ant = np.array([rho_ant**i for i in range(N_a)])
    Sigma_ant = scipy.linalg.toeplitz(sigma_ant)  # (N_a, N_a), diag = ones
    
    sigma_car = np.array([rho_car**j for j in range(N_c)])
    Sigma_car = scipy.linalg.toeplitz(sigma_car)  # (N_c, N_c), diag = ones
    
    # Kronecker product
    Sigma_base = np.kron(Sigma_ant, Sigma_car)  # (N_a*N_c, N_a*N_c)
    
    # Scale so diagonal entries = N_a*N_c
    scale = N_a * N_c
    Sigma = scale * Sigma_base  # diag(Σ) = [N_a*N_c, ..., N_a*N_c]
    
    # Cholesky decomposition
    L = np.linalg.cholesky(Sigma)
    
    # Sample: h = L @ z where z ~ CN(0, I)
    z = (np.random.randn(N_a * N_c) + 1j * np.random.randn(N_a * N_c)) / np.sqrt(2)
    h = L @ z
    
    # Reshape to matrix
    H = h.reshape(N_a, N_c)
    
    return torch.tensor(H, device=device, dtype=torch.complex64), torch.tensor(Sigma, device=device, dtype=torch.complex64)


def generate_gaussian_spikes_channel(N_a, N_c, spike_indices=None, spike_amplitude=2.0, spike_width=2, device='cuda'):
    """
    Generate channel with Gaussian spikes at specific subcarriers.
    
    Model: H[a,k] ~ CN(μ_k, σ_k^2)
    where:
        - μ_k: Mean at subcarrier k (has spikes at specified indices)
        - σ_k^2: Variance at subcarrier k (constant across antennas, varies across subcarriers)
    
    The covariance is block-diagonal: independent across subcarriers, shared across antennas.
    
    This is perfect for testing algorithm's ability to:
    - Track energy spikes in frequency domain
    - Adjust variance based on subcarrier importance
    - Handle structured uncertainty (different spikes have different variances)
    
    Args:
        N_a: Number of antennas
        N_c: Number of subcarriers
        spike_indices: List of subcarrier indices where to place high-energy spikes (default: [16, 32, 48])
        spike_amplitude: Amplitude of spike mean (default: 2.0)
        spike_width: Gaussian envelope width (std dev in subcarriers) (default: 2)
        device: cuda/cpu
    
    Returns:
        H: Channel matrix (N_a, N_c) [complex]
        Sigma: Block-diagonal covariance (N_a*N_c, N_a*N_c) [complex]
    """
    if spike_indices is None:
        spike_indices = [16, 32, 48]
    
    # Create mean vector: Gaussian spikes at specified frequencies
    mu_k = np.zeros(N_c, dtype=complex)
    for spike_idx in spike_indices:
        # Gaussian envelope centered at spike_idx
        for k in range(N_c):
            gaussian_val = np.exp(-(k - spike_idx)**2 / (2 * spike_width**2))
            mu_k[k] += spike_amplitude * gaussian_val * np.exp(1j * 0.2 * np.pi)  # Add phase
    
    # Normalize mean to get reasonable magnitude
    max_mean = np.max(np.abs(mu_k))
    if max_mean > 0:
        mu_k = mu_k * (N_a * N_c) / (np.linalg.norm(mu_k) + 1e-8)
    
    # Create variance profile: higher variance at spikes, lower elsewhere
    sigma_k_sq = 2.0 * np.ones(N_c, dtype=float)  # Baseline variance = 2.0
    for spike_idx in spike_indices:
        for k in range(N_c):
            gaussian_env = np.exp(-(k - spike_idx)**2 / (2 * (spike_width + 1)**2))
            sigma_k_sq[k] += 4.0 * gaussian_env  # Spikes: 2.0 + 4.0 = 6.0 (3x contrast)
    
    # Build block-diagonal covariance: Σ = block_diag(Σ_0, Σ_1, ..., Σ_{N_c-1})
    # where Σ_k = σ_k^2 * I_{N_a} (identity across antennas, independent subcarriers)
    Sigma = np.zeros((N_a * N_c, N_a * N_c), dtype=complex)
    for k in range(N_c):
        # Each block is σ_k^2 * I_{N_a}
        Sigma[k * N_a : (k + 1) * N_a, k * N_a : (k + 1) * N_a] = sigma_k_sq[k] * np.eye(N_a)
    
    # Sample: H[a,k] ~ CN(μ_k, σ_k^2)
    H = np.zeros((N_a, N_c), dtype=complex)
    for k in range(N_c):
        # Generate N_a complex Gaussian samples with mean μ_k and variance σ_k^2
        noise = np.sqrt(sigma_k_sq[k]) * (np.random.randn(N_a) + 1j * np.random.randn(N_a)) / np.sqrt(2)
        H[:, k] = mu_k[k] + noise
    
    # Verify norm is approximately N_a*N_c
    current_norm = np.linalg.norm(H)
    scaling = (N_a * N_c) / (current_norm + 1e-8)
    H = scaling * H
    Sigma = (scaling**2) * Sigma
    
    return torch.tensor(H, device=device, dtype=torch.complex64), torch.tensor(Sigma, device=device, dtype=torch.complex64)


def compute_score(h, Sigma_inv):
    """
    Compute score ∇log p(h) for Gaussian prior.
    
    For p(h) ~ CN(0, Σ), the score is:
    ∇log p(h) = -Σ^(-1) @ h
    
    Args:
        h: Channel vector (N_a*N_c,) [complex]
        Sigma_inv: Inverse of covariance matrix (N_a*N_c, N_a*N_c) [complex]
    
    Returns:
        score: -Σ^(-1) @ h
    """
    h_vec = h.reshape(-1)  # Vectorize if matrix
    score = -Sigma_inv @ h_vec
    return score.reshape(h.shape)


def denoise_step_tweedie_with_score(h_t, score_t, alphabar_t, alphabar_t_next):
    """
    Denoise using Tweedie formula + analytical score.
    
    Instead of: H̃ = f_θ(H^t, t)
    We use:     H̃ = (1/√ᾱ_t) * (H^t + (1-ᾱ_t) * score)
    
    Then apply scheduler update.
    
    Args:
        h_t: Current noisy sample (N_a, N_c) [complex]
        score_t: True score at time t (N_a, N_c) [complex]
        alphabar_t: ᾱ_t (scalar)
        alphabar_t_next: ᾱ_t_next (scalar)
    
    Returns:
        h_t_next: Denoised sample at next timestep
    """
    sqrt_alphabar_t = torch.sqrt(alphabar_t)
    sqrt_1_minus_alphabar_t = torch.sqrt(1 - alphabar_t)
    sqrt_alphabar_t_next = torch.sqrt(alphabar_t_next)
    sqrt_1_minus_alphabar_t_next = torch.sqrt(1 - alphabar_t_next)
    
    # Step 1: Tweedie prediction of x_0
    h_0_pred = (1 / sqrt_alphabar_t) * (h_t + (1 - alphabar_t) * score_t)
    
    # Step 2: DDIM step (reverse diffusion formula)
    # h_t_next = sqrt(ᾱ_next) * h_0_pred + sqrt(1-ᾱ_next) * noise_pred
    noise_pred = (h_t - sqrt_alphabar_t * h_0_pred) / sqrt_1_minus_alphabar_t
    h_t_next = sqrt_alphabar_t_next * h_0_pred + sqrt_1_minus_alphabar_t_next * noise_pred
    
    return h_t_next


def apply_conditioning_gradient(h, Y, x_tau, sigma_H, xi=0.1, num_steps=5):
    """
    Apply gradient descent to pull h toward channels that explain Y.
    
    Minimizes: ||Y_τ - σ̂_H * H @ diag(x_τ)||²_F
    where diag(x_tau) is a (N_c, N_c) diagonal matrix with x_tau on diagonal.
    
    Args:
        h: Current channel (N_a, N_c) [complex]
        Y: Received signal (N_a, N_c) [complex]
        x_tau: Transmitted symbols (N_c,) [complex], with pilots=1.0 and data=QPSK
        sigma_H: Channel power normalization (scalar)
        xi: Step size
        num_steps: Number of gradient steps
    
    Returns:
        h_guided: Updated channel after gradient steps
    """
    h_guided = h.clone().detach().requires_grad_(True)
    N_c = h.shape[1]
    
    for _ in range(num_steps):
        # Compute loss: ||Y - σ_H * H @ diag(x)||²_F
        # H @ diag(x): each column j of H is scaled by x[j]
        x_diag = torch.diag(x_tau)  # (N_c, N_c) diagonal matrix
        H_scaled = sigma_H * h_guided @ x_diag  # (N_a, N_c)
        loss = torch.sum(torch.abs(Y - H_scaled)**2)
        
        # Gradient step
        loss.backward()
        with torch.no_grad():
            h_guided.data -= xi * h_guided.grad
            h_guided.grad.zero_()
    
    return h_guided.detach()


def compute_disagreement(particles, data_indices):
    """
    Compute disagreement (entropy) between particles for each subcarrier.
    
    Uses log-sum-exp trick for numerical stability:
    entropy[j] = ∑_i log(∑_i' exp(||H̃_i[:,j] - H̃_i'[:,j]||²_F))
    
    This measures the entropy (uncertainty) of particle disagreement at each subcarrier,
    which is more theoretically accurate than simple pairwise sum.
    
    Args:
        particles: (N_p, N_a, N_c) - All particles [complex]
        data_indices: List of subcarrier indices in data region
    
    Returns:
        disagreement: (len(data_indices),) - Entropy per subcarrier
    """
    N_p = particles.shape[0]
    N_a = particles.shape[1]
    disagreement = torch.zeros(len(data_indices), device=particles.device, dtype=torch.float32)
    
    for j_idx, j in enumerate(data_indices):
        # Extract column j from all particles: (N_p, N_a)
        h_j_all = particles[:, :, j]
        
        # Compute entropy at subcarrier j using log-sum-exp
        entropy_j = 0.0
        for i in range(N_p):
            # Compute squared distances from particle i to all other particles
            # distances_i[i'] = ||H̃_i[:,j] - H̃_i'[:,j]||²_F (Frobenius norm squared)
            distances_i = torch.zeros(N_p, device=particles.device, dtype=torch.float32)
            for i_prime in range(N_p):
                diff = h_j_all[i] - h_j_all[i_prime]
                distances_i[i_prime] = torch.sum(torch.abs(diff)**2)
            
            # Log-sum-exp: log(∑_i' exp(distances_i[i']))
            # torch.logsumexp handles numerical stability automatically
            lse = torch.logsumexp(distances_i, dim=0)
            entropy_j = entropy_j + lse
        
        disagreement[j_idx] = entropy_j
    
    return disagreement


def initialize_params(h_true, N_p, gamma=14.4, sigma_n_sq=0.01, alphas_cumprod=None, device='cuda'):
    """
    Initialize sampling parameters: signal generation, hyperparameters, and OFDM symbols.
    
    Generates Y_0 = sigma_H_true * H @ diag(x_tau) + noise with proper signal model.
    Computes sigma_H, delta, t_0 from received signal.
    
    Args:
        h_true: Ground truth channel (N_a, N_c) [complex]
        N_p: Number of particles (used to compute ratio, but not for initialization)
        gamma: Initialization strength
        sigma_n_sq: Noise power from training
        alphas_cumprod: Cumulative product of alphas from diffusion schedule
        device: cuda/cpu
    
    Returns:
        sigma_H: Channel power estimate from Y_0
        delta: Noise scaling factor
        t_0: Initial timestep
        Y_0: Received signal (N_a, N_c) [complex]
        x_tau: Transmitted OFDM symbols (N_c,) [complex]
        pilot_indices: Indices of initial pilots
        data_indices: Indices of data region
    """
    N_a, N_c = h_true.shape
    
    # Estimate true channel gain
    sigma_H_true = torch.norm(h_true, p='fro') / np.sqrt(N_a * N_c)
    
    # Pilot/data split
    num_pilots = 2
    num_data = N_c - num_pilots
    ratio = num_pilots / (num_pilots + num_data)
    
    all_sc_indices = torch.arange(N_c, device=device)
    pilot_indices = torch.randperm(N_c, device=device)[:num_pilots]
    data_indices = all_sc_indices[torch.isin(all_sc_indices, pilot_indices, invert=True)]
    
    # Initialize x_tau: pilots transmit 1, data subcarriers transmit random QPSK
    # x_tau is now (N_c,) - one symbol per subcarrier for all antennas
    x_tau = torch.ones((N_c,), dtype=torch.complex64, device=device)
    x_tau[pilot_indices] = 1.0
    
    qpsk_symbols = torch.tensor([1.0, -1.0, 1.0j, -1.0j], dtype=torch.complex64, device=device)
    for d_j in data_indices:
        random_qpsk = qpsk_symbols[torch.randint(0, 4, (1,), device=device)]
        x_tau[d_j] = random_qpsk
    
    # Generate Y_0 with proper signal model: Y_0 = sigma_H_true * H @ diag(x_tau) + noise
    # Y[a,j] = sigma_H_true * H[a,j] * x_tau[j] + noise[a,j]
    noise = (torch.randn_like(h_true) + 1j * torch.randn_like(h_true)) / np.sqrt(2)
    x_diag = torch.diag(x_tau)  # (N_c, N_c) diagonal matrix
    Y_0 = sigma_H_true * h_true @ x_diag + 0.1 * noise
    
    # Estimate sigma_H from received signal Y_0
    sigma_H = torch.norm(Y_0, p='fro') / np.sqrt(N_a * N_c)
    
    # Compute delta: delta = sqrt(1 - gamma^2 * (1 + sigma_n^2/sigma_H^2) * |P|/(|P|+|D|))
    term = gamma**2 * (1 + sigma_n_sq / (sigma_H**2 + 1e-8)) * ratio
    delta = torch.sqrt(torch.clamp(1 - term, min=0))
    
    # Compute t_0: argmin_t {alphabar_t - gamma * (|P|/(|P|+|D|))}
    if alphas_cumprod is not None:
        target = gamma * ratio
        distances = torch.abs(alphas_cumprod - target)
        t_0 = torch.argmin(distances).item()
    else:
        t_0 = int(gamma * ratio * 999)
    
    return sigma_H, delta, t_0, Y_0, x_tau, pilot_indices, data_indices


def initialize_particles(Y_0, sigma_H, delta, t_0, pilot_indices, data_indices, N_p, device='cuda'):
    """
    Initialize N_p particles at timestep t_0.
    
    Uses the matrix signal model: Y = sigma_H * H @ diag(x) + noise
    where x_tau[j]=1 at pilots, so Y[a,j] = sigma_H * H[a,j] + noise at pilots.
    
    Args:
        Y_0: Received signal (N_a, N_c) [complex]
        sigma_H: Channel power estimate (scalar)
        delta: Noise scaling factor (scalar)
        t_0: Starting timestep
        pilot_indices: Indices of pilot subcarriers
        data_indices: Indices of data subcarriers
        N_p: Number of particles
        device: cuda/cpu
    
    Returns:
        particles: (N_p, N_a, N_c) initialized particles [complex]
    """
    N_a, N_c = Y_0.shape
    gamma = 14.4  # Match initialization strength
    
    particles = torch.zeros((N_p, N_a, N_c), dtype=torch.complex64, device=device)
    
    for i in range(N_p):
        h_i = torch.zeros_like(particles[i])
        
        # At pilot subcarriers (x_tau[j]=1): Y[a,j] = sigma_H * H[a,j] + noise
        # So H[a,j] ≈ gamma * (Y[a,j] / sigma_H) + delta * epsilon
        for p_j in pilot_indices:
            for ant in range(N_a):
                y_pn = Y_0[ant, p_j]
                epsilon_pn = (torch.randn(1, dtype=torch.complex64, device=device) + 
                             1j * torch.randn(1, dtype=torch.complex64, device=device)) / np.sqrt(2)
                h_i[ant, p_j] = gamma * (y_pn / (sigma_H + 1e-8)) + delta * epsilon_pn
        
        # At data subcarriers: no direct observation (x_tau[j]∈{QPSK})
        # Initialize with noise only: h = delta * epsilon
        for d_j in data_indices:
            for ant in range(N_a):
                epsilon_dn = (torch.randn(1, dtype=torch.complex64, device=device) + 
                             1j * torch.randn(1, dtype=torch.complex64, device=device)) / np.sqrt(2)
                h_i[ant, d_j] = delta * epsilon_dn
        
        particles[i] = h_i
    
    return particles





def create_diffusion_schedule(num_train_timesteps=1000, beta_min=0.0001, beta_max=0.02, device='cuda'):
    """
    Create linear diffusion schedule: β_t linearly spaced from beta_min to beta_max.
    
    Returns alphas and alphas_cumprod for the full schedule.
    
    Args:
        num_train_timesteps: Total training timesteps (T)
        beta_min: Minimum beta value at t=0 (default: 0.0001)
        beta_max: Maximum beta value at t=T (default: 0.02)
        device: cuda/cpu
    
    Returns:
        alphas: (num_train_timesteps,)
        alphas_cumprod: (num_train_timesteps,)
    """
    # Create linear schedule: β_t linearly spaced from beta_min to beta_max
    betas = torch.linspace(beta_min, beta_max, num_train_timesteps, dtype=torch.float32, device=device)
    
    # Compute alphas and cumulative products
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return alphas, alphas_cumprod


def sample_gaussian_posterior(Y_0, Sigma, Sigma_inv, pilot_indices, data_indices, N_p=8, N_gen=500, xi=0.1, 
                              gamma=14.4, sigma_n_sq=0.01, selection_freq=50, device='cuda'):
    """
    Main active sampling loop for Gaussian prior with true score.
    
    Signal model: Y = sigma_H * H @ diag(x_tau) + noise
    where x_tau is (N_c,) with x_tau[j]=1.0 at pilot indices and QPSK symbols at data indices.
    
    Args:
        Y_0: Initial received signal (N_a, N_c)
        Sigma: Covariance matrix (N_a*N_c, N_a*N_c)
        Sigma_inv: Inverse covariance
        pilot_indices: Indices of pilot subcarriers
        data_indices: Indices of data subcarriers
        N_p: Number of particles
        N_gen: Number of denoising steps
        xi: Guidance step size
        gamma: Initialization strength
        sigma_n_sq: Noise power from training
        selection_freq: Frequency (in steps) for active pilot selection (default: 50)
        device: cuda/cpu
    
    Returns:
        particles_final: (N_p, N_a, N_c) - Final channel estimates
        history: Dictionary with sampling history
    """
    N_a, N_c = Y_0.shape
    history = {
        'pilots_selected': [],
        'disagreements': [],
        'particle_norms': [],
        'sigma_H_values': []  # Track adaptive sigma_H evolution
    }
    
    # Create diffusion schedule (for alphas and alphas_cumprod)
    alphas, alphas_cumprod = create_diffusion_schedule(
        num_train_timesteps=1000, 
        beta_min=0.0001,
        beta_max=0.02,
        device=device
    )
    
    N_a, N_c = Y_0.shape
    
    # Compute sigma_H from Y_0
    sigma_H = torch.norm(Y_0, p='fro') / np.sqrt(N_a * N_c)
    
    # Compute delta and t_0
    num_pilots = len(pilot_indices)
    num_data = len(data_indices)
    ratio = num_pilots / (num_pilots + num_data)
    term = gamma**2 * (1 + sigma_n_sq / (sigma_H**2 + 1e-8)) * ratio
    delta = torch.sqrt(torch.clamp(1 - term, min=0))
    
    # Compute t_0
    target = gamma * ratio
    distances = torch.abs(alphas_cumprod - target)
    t_0 = torch.argmin(distances).item()
    
    # Initialize particles at t_0
    particles = initialize_particles(Y_0, sigma_H, delta, t_0, pilot_indices, 
                                      data_indices, N_p, device)
    
    # Initialize x_tau: (N_c,) vector with pilots=1.0, data=QPSK
    x_tau = torch.ones((N_c,), dtype=torch.complex64, device=device)
    x_tau[pilot_indices] = 1.0
    
    qpsk_symbols = torch.tensor([1.0, -1.0, 1.0j, -1.0j], dtype=torch.complex64, device=device)
    for d_j in data_indices:
        random_qpsk = qpsk_symbols[torch.randint(0, 4, (1,), device=device)]
        x_tau[d_j] = random_qpsk
    
    Y_tau = Y_0.clone()
    tau = 0
    
    print(f"Starting active diffusion sampling:")
    print(f"  N_p={N_p}, N_gen={N_gen}, xi={xi}, gamma={gamma}")
    print(f"  sigma_H={sigma_H:.4f}, t_0={t_0}, delta={delta:.4f}")
    print(f"  Pilot indices: {pilot_indices.tolist()}")
    print(f"  Active selection frequency: every {selection_freq} steps")
    
    # Main denoising loop: while t > 0
    t = t_0
    step_count = 0
    with tqdm(total=N_gen, desc="DDIM Sampling") as pbar:
        while t > 0:
            # Compute next timestep: t_next = t - round(t_0 / N_gen)
            t_next = t - round(t_0 / N_gen)
            t_next = max(0, t_next)  # Ensure non-negative
            
            # Get alphabars for current and next timestep
            alphabar_t = alphas_cumprod[t]
            alphabar_t_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            
            # Denoise each particle
            for i in range(N_p):
                # Compute score for current particle
                score = compute_score(particles[i], Sigma_inv)
                
                # Denoise step using Tweedie + score
                particles[i] = denoise_step_tweedie_with_score(
                    particles[i], score, alphabar_t, alphabar_t_next
                )
                
                # Conditioning gradient: pull toward Y_tau
                particles[i] = apply_conditioning_gradient(
                    particles[i], Y_tau, x_tau, sigma_H, xi=xi, num_steps=3
                )
                
                # Power normalization
                norm_i = torch.norm(particles[i], p='fro')
                particles[i] = particles[i] / (norm_i + 1e-8)
            
            # Active selection at specified timesteps
            if step_count % selection_freq == 0 and step_count > 0:
                tau += 1
                
                # Compute disagreement for data subcarriers
                disagreement = compute_disagreement(particles, data_indices)
                
                # Select most disagreed-upon subcarrier
                j_star = data_indices[torch.argmax(disagreement)].item()
                
                # Update x_tau: set the newly selected subcarrier to pilot (transmit 1)
                x_tau[j_star] = 1.0
                
                # Simulate pilot observation at subcarrier j*
                # Y[a,j*] = sigma_H * H[a,j*] + noise (since x_tau[j*]=1)
                noise_tau = (torch.randn((N_a,), dtype=torch.complex64, device=device) + 
                            1j * torch.randn((N_a,), dtype=torch.complex64, device=device)) / np.sqrt(2)
                y_pilot = sigma_H * particles[0, :, j_star] + noise_tau
                Y_tau[:, j_star] = y_pilot
                
                # Update sigma_H adaptively based on new Y_tau
                sigma_H = torch.norm(Y_tau, p='fro') / np.sqrt(N_a * N_c)
                
                history['pilots_selected'].append(j_star)
                history['sigma_H_values'].append(sigma_H.item())
                history['disagreements'].append(disagreement.cpu().numpy())
                
                print(f"  τ={tau}: Selected pilot at subcarrier {j_star}, max disagreement={disagreement.max():.4f}, sigma_H={sigma_H:.4f}")
            
            # Move to next timestep
            t = t_next
            step_count += 1
            pbar.update(1)
    
    # Record final norms
    for i in range(N_p):
        history['particle_norms'].append(torch.norm(particles[i], p='fro').item())
    
    return particles, history


if __name__ == "__main__":
    # Test parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_a, N_c = 32, 64  # Match original model dimensions
    N_p = 4  # Small for testing
    N_gen = 100  # Reduced for quick test
    
    print("=" * 80)
    print("Testing Active Diffusion Sampling with Gaussian Prior")
    print("=" * 80)
    
    # Generate ground truth channel and covariance
    print("[0] Generating structured Gaussian channel...")
    h_true, Sigma = generate_structured_gaussian_channel(N_a, N_c, rho_ant=0.7, rho_car=0.7, device=device)
    Sigma_inv = torch.linalg.inv(Sigma)
    print(f"    Channel shape: {h_true.shape}")
    print(f"    Covariance shape: {Sigma.shape}")
    print(f"    Channel norm: {torch.norm(h_true, p='fro'):.4f}")
    
    # Create diffusion schedule (for alphas and alphas_cumprod)
    alphas, alphas_cumprod = create_diffusion_schedule(
        num_train_timesteps=1000, 
        beta_min=0.0001,
        beta_max=0.02,
        device=device
    )
    
    # Initialize parameters: signal, hyperparameters, OFDM symbols
    sigma_H, delta, t_0, Y_0, x_tau, pilot_indices, data_indices = \
        initialize_params(h_true, N_p, gamma=14.4, sigma_n_sq=0.01, 
                         alphas_cumprod=alphas_cumprod, device=device)
    
    # Run active sampling
    print("\n[2] Running active diffusion sampling...")
    particles, history = sample_gaussian_posterior(
        Y_0, Sigma, Sigma_inv,
        pilot_indices, data_indices,
        N_p=N_p,
        N_gen=N_gen,
        xi=0.1,
        gamma=14.4,
        sigma_n_sq=0.01,
        device=device
    )
    
    print(f"\n[3] Results:")
    print(f"    Final particles shape: {particles.shape}")
    for i in range(N_p):
        error = torch.norm(particles[i] - h_true, p='fro') / torch.norm(h_true, p='fro')
        print(f"    Particle {i}: norm={history['particle_norms'][i]:.4f}, rel_error={error:.4f}")
    
    print(f"\n    Pilots selected during active sampling: {history['pilots_selected']}")
    print(f"    Number of active selection steps: {len(history['pilots_selected'])}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)