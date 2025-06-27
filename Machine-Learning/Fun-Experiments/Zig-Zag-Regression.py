import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset (linear + sinusoidal + noise)
np.random.seed(42)
x_np = np.linspace(0, 10, 200)
y_true = 2 * x_np + 1
noise = np.random.normal(0, 1.5, size=x_np.shape)
y_np = y_true + 2 * np.sin(np.pi * x_np) + noise

# Convert to PyTorch tensors
x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# Define Enhanced Neural Zigzagression Model
class NeuralZigzagression(nn.Module):
    def __init__(self, n_segments=10, omega_init=np.pi, segment_wise_omega=True, 
                 learnable_segments=False, add_phase=True):
        super(NeuralZigzagression, self).__init__()
        self.n_segments = n_segments
        self.segment_wise_omega = segment_wise_omega
        self.learnable_segments = learnable_segments
        self.add_phase = add_phase
        
        # Segment boundaries
        if learnable_segments:
            # Initialize with equal spacing but make them learnable
            init_bounds = torch.linspace(x.min(), x.max(), n_segments + 1)
            # Only interior boundaries are learnable (keep endpoints fixed)
            self.segment_bounds_learnable = nn.Parameter(init_bounds[1:-1])
            self.x_min = x.min()
            self.x_max = x.max()
        else:
            self.segment_bounds = torch.linspace(x.min(), x.max(), n_segments + 1)
        
        # Parameters
        self.radius = nn.Parameter(torch.ones(n_segments))  # Learnable radii
        self.m = nn.Parameter(torch.tensor(1.0))  # Learnable slope
        self.b = nn.Parameter(torch.tensor(0.0))  # Learnable intercept
        
        # Frequency parameters
        if segment_wise_omega:
            self.omega = nn.Parameter(torch.ones(n_segments) * omega_init)  # Per-segment omega
        else:
            self.omega = nn.Parameter(torch.tensor(omega_init))  # Global omega
        
        # Phase parameters
        if add_phase:
            self.phi = nn.Parameter(torch.zeros(n_segments))  # Learnable phase shifts

    def get_segment_bounds(self):
        if self.learnable_segments:
            # Combine fixed endpoints with learnable interior points
            return torch.cat([
                torch.tensor([self.x_min]), 
                torch.sort(self.segment_bounds_learnable)[0],  # Sort to maintain order
                torch.tensor([self.x_max])
            ])
        else:
            return self.segment_bounds

    def forward(self, x):
        y_pred = self.m * x + self.b
        segment_bounds = self.get_segment_bounds()
        
        for i in range(self.n_segments):
            left = segment_bounds[i]
            right = segment_bounds[i + 1]
            mask = (x >= left) & (x < right)
            radius = self.radius[i]
            
            # Get omega for this segment
            if self.segment_wise_omega:
                omega_i = self.omega[i]
            else:
                omega_i = self.omega
            
            # Get phase for this segment
            if self.add_phase:
                phase_i = self.phi[i]
                y_pred += mask.float() * (radius * torch.sin(omega_i * x + phase_i))
            else:
                y_pred += mask.float() * (radius * torch.sin(omega_i * x))
                
        return y_pred

# Initialize enhanced model
model = NeuralZigzagression(
    n_segments=10, 
    omega_init=np.pi, 
    segment_wise_omega=True,  # Try per-segment frequencies
    learnable_segments=False,  # Keep segment boundaries fixed for now
    add_phase=True  # Add phase shifts
)

# Add L2 regularization to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# Train the model
loss_history = []
omega_history = []
radius_history = []
for epoch in range(1500):  # More epochs for complex model
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    
    # Main loss + regularization on radii to encourage sparsity
    main_loss = loss_fn(y_pred, y)
    radius_reg = 0.01 * torch.mean(torch.abs(model.radius))  # L1 on radii
    loss = main_loss + radius_reg
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(main_loss.item())  # Track main loss only
    if model.segment_wise_omega:
        omega_history.append(model.omega.mean().item())  # Mean omega across segments
    else:
        omega_history.append(model.omega.item())
    radius_history.append(model.radius.mean().item())

# Print final learned parameters
print("=== LEARNED PARAMETERS ===")
if model.segment_wise_omega:
    print(f"Learned omegas: {model.omega.detach().numpy()}")
    print(f"Mean omega: {model.omega.mean().item():.4f} (true: {np.pi:.4f})")
else:
    print(f"Learned omega: {model.omega.item():.4f} (true: {np.pi:.4f})")
print(f"Learned slope: {model.m.item():.4f} (true: 2.0)")
print(f"Learned intercept: {model.b.item():.4f} (true: 1.0)")
print(f"Learned radii: {model.radius.detach().numpy()}")
if model.add_phase:
    print(f"Learned phases: {model.phi.detach().numpy()}")
print(f"Final loss: {loss_history[-1]:.6f}")

# Plot comprehensive results
x_plot = x.detach().numpy().squeeze()
y_plot = y.detach().numpy().squeeze()
y_pred_plot = model(x).detach().numpy().squeeze()

plt.figure(figsize=(18, 12))

# Main fit plot
plt.subplot(2, 3, 1)
plt.scatter(x_plot, y_plot, label='Noisy Data', alpha=0.4, s=20)
plt.plot(x_plot, y_pred_plot, color='red', label='Neural Zigzagression', linewidth=3)
plt.plot(x_plot, y_true, color='green', linestyle='--', alpha=0.7, label='True Linear Trend')
plt.title("Enhanced Neural Zigzagression Fit", fontsize=14)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)

# Loss evolution
plt.subplot(2, 3, 2)
plt.plot(loss_history, color='blue', linewidth=2)
plt.title("Training Loss Evolution", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Omega evolution
plt.subplot(2, 3, 3)
plt.plot(omega_history, color='purple', linewidth=2)
plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'True omega ({np.pi:.4f})')
plt.title("Omega Learning Progress", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Omega Value")
plt.legend()
plt.grid(True, alpha=0.3)

# Learned parameters per segment
plt.subplot(2, 3, 4)
x_pos = np.arange(model.n_segments)
width = 0.35
plt.bar(x_pos - width/2, model.radius.detach().numpy(), width, 
        label='Radius', alpha=0.8, color='orange')
if model.segment_wise_omega:
    plt.bar(x_pos + width/2, model.omega.detach().numpy(), width, 
            label='Omega', alpha=0.8, color='green')
plt.title("Learned Parameters per Segment", fontsize=14)
plt.xlabel("Segment")
plt.ylabel("Parameter Value")
plt.legend()
plt.grid(True, alpha=0.3)

# Phase parameters (if enabled)
if model.add_phase:
    plt.subplot(2, 3, 5)
    plt.bar(range(model.n_segments), model.phi.detach().numpy(), 
            color='cyan', alpha=0.8)
    plt.title("Learned Phase Shifts", fontsize=14)
    plt.xlabel("Segment")
    plt.ylabel("Phase (radians)")
    plt.grid(True, alpha=0.3)

# Residuals analysis
plt.subplot(2, 3, 6)
residuals = y_plot - y_pred_plot
plt.scatter(x_plot, residuals, alpha=0.6, s=20)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Residuals Analysis", fontsize=14)
plt.xlabel("x")
plt.ylabel("Residuals")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis
print("\n=== MODEL ANALYSIS ===")
print(f"R² Score: {1 - np.var(residuals) / np.var(y_plot):.4f}")
print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")

# Show segment boundaries and their characteristics
segment_bounds = model.get_segment_bounds().detach().numpy()
print(f"\nSegment boundaries: {segment_bounds}")
print(f"Segment widths: {np.diff(segment_bounds)}")
