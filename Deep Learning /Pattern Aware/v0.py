import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic house price data (replace with real dataset)
n_samples = 1000
n_features = 13  # e.g., rooms, area, etc.
X = np.random.randn(n_samples, n_features)  # Features
y = np.sum(X * np.random.randn(n_features), axis=1) + np.random.randn(n_samples) * 0.1  # Prices
y = y.reshape(-1, 1)

# Preprocess data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y)

# Define the ANN model
class HousePriceANN(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceANN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

class PatternMemory:
    """Advanced pattern memory system"""
    def __init__(self, pattern_dim, convergence_threshold=0.01, similarity_threshold=0.8):
        self.learned_patterns = []  # Store learned patterns
        self.pattern_errors = []    # Track error for each pattern
        self.convergence_threshold = convergence_threshold
        self.similarity_threshold = similarity_threshold
        self.pattern_history = defaultdict(list)
    
    def compute_pattern_signature(self, activations, weights):
        """Compute a signature for the current network state"""
        # Combine activation patterns and weight patterns
        act_signature = torch.mean(activations, dim=1)  # Average activation per neuron
        weight_signature = torch.norm(weights, dim=1)   # Weight magnitude per neuron
        
        # Normalize signatures
        act_signature = act_signature / (torch.norm(act_signature) + 1e-8)
        weight_signature = weight_signature / (torch.norm(weight_signature) + 1e-8)
        
        return torch.cat([act_signature, weight_signature])
    
    def is_pattern_learned(self, pattern_signature, current_error):
        """Check if current pattern is already learned"""
        if len(self.learned_patterns) == 0:
            return False, -1
        
        # Convert to numpy for easier computation
        current_sig = pattern_signature.detach().numpy()
        
        for i, (learned_sig, learned_error) in enumerate(zip(self.learned_patterns, self.pattern_errors)):
            # Compute cosine similarity
            similarity = np.dot(current_sig, learned_sig) / (
                np.linalg.norm(current_sig) * np.linalg.norm(learned_sig) + 1e-8
            )
            
            # Check if pattern is similar and error is converged
            if (similarity > self.similarity_threshold and 
                abs(current_error - learned_error) < self.convergence_threshold):
                return True, i
        
        return False, -1
    
    def add_pattern(self, pattern_signature, error):
        """Add a new learned pattern"""
        self.learned_patterns.append(pattern_signature.detach().numpy())
        self.pattern_errors.append(error)
    
    def update_pattern(self, pattern_idx, error):
        """Update existing pattern's error"""
        self.pattern_errors[pattern_idx] = error

def compute_adaptive_pattern_matrix(activations, weights, input_data):
    """
    Compute pattern matrix focusing on neuron specialization and input sensitivity
    """
    # Compute neuron specialization (how selective each neuron is)
    activation_std = torch.std(activations, dim=1, keepdim=True)  # (64, 1)
    activation_mean = torch.mean(activations, dim=1, keepdim=True)  # (64, 1)
    
    # Neurons with high std are more specialized
    specialization_score = activation_std / (activation_mean + 1e-8)  # (64, 1)
    
    # Compute input sensitivity (which inputs activate which neurons most)
    input_sensitivity = torch.abs(torch.mm(weights, input_data))  # (64, 1000)
    
    # Combine specialization and sensitivity
    pattern_matrix = specialization_score * input_sensitivity  # (64, 1000)
    
    # Normalize to get probabilities
    pattern_matrix = torch.softmax(pattern_matrix / 0.1, dim=0)  # Temperature = 0.1
    
    return pattern_matrix

def train_true_pattern_aware(model, X, y, epochs=200, lr=0.001, pattern_freq=10):
    """Training with true pattern awareness"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize pattern memory
    pattern_memory = PatternMemory(pattern_dim=128)  # 64 activations + 64 weights
    
    losses = []
    pattern_learned_flags = []
    suppression_strengths = []
    learning_rates = []
    
    base_lr = lr
    
    for epoch in range(epochs):
        # Compute current loss for pattern analysis
        with torch.no_grad():
            current_outputs = model(X)
            current_loss = criterion(current_outputs, y).item()
        
        # Pattern analysis every pattern_freq epochs
        if epoch % pattern_freq == 0:
            # Get current network state
            with torch.no_grad():
                activations = model.layer1(X).t()  # (64, 1000)
                weights = model.layer1.weight.data  # (64, 13)
            
            # Compute pattern signature
            pattern_signature = pattern_memory.compute_pattern_signature(activations, weights)
            
            # Check if this pattern is already learned
            is_learned, pattern_idx = pattern_memory.is_pattern_learned(pattern_signature, current_loss)
            
            if is_learned:
                # Pattern is learned - suppress training on it
                suppression_strength = min(0.9, 0.1 + (epoch / epochs) * 0.8)  # Increase over time
                adjusted_lr = base_lr * (1 - suppression_strength)
                
                # Compute which neurons/weights contribute most to this learned pattern
                pattern_matrix = compute_adaptive_pattern_matrix(activations, weights, X.t())
                
                # Create suppression mask
                neuron_importance = torch.mean(pattern_matrix, dim=1)  # (64,)
                weight_mask = 1 - (neuron_importance.unsqueeze(1) * suppression_strength)  # (64, 1)
                weight_mask = weight_mask.expand(-1, weights.size(1))  # (64, 13)
                
                print(f"Epoch {epoch}: LEARNED PATTERN DETECTED (#{pattern_idx}) - Suppressing {suppression_strength:.2f}")
                
                # Update pattern memory
                pattern_memory.update_pattern(pattern_idx, current_loss)
                
            else:
                # New pattern - normal learning
                suppression_strength = 0.0
                adjusted_lr = base_lr
                weight_mask = torch.ones_like(weights)
                
                # Add to pattern memory if loss is low enough (pattern seems to be learning)
                if len(pattern_memory.learned_patterns) < 10:  # Limit memory size
                    pattern_memory.add_pattern(pattern_signature, current_loss)
                    print(f"Epoch {epoch}: New pattern added to memory (Total: {len(pattern_memory.learned_patterns)})")
            
            pattern_learned_flags.append(is_learned)
            suppression_strengths.append(suppression_strength)
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr
            learning_rates.append(adjusted_lr)
        
        # Standard training step
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Apply suppression mask if pattern was detected as learned
        if epoch % pattern_freq == 0 and 'weight_mask' in locals():
            if model.layer1.weight.grad is not None:
                model.layer1.weight.grad *= weight_mask
        
        optimizer.step()
        losses.append(loss.item())
        
        # Regular progress reporting
        if epoch % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
    
    return losses, pattern_learned_flags, suppression_strengths, learning_rates

# Train the model with true pattern awareness
print("Training with True Pattern-Aware Learning...")
model = HousePriceANN(input_dim=n_features)
losses, pattern_flags, suppressions, lrs = train_true_pattern_aware(
    model, X_train, y_train, epochs=200, lr=0.001, pattern_freq=10
)

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Training loss
ax1.plot(losses, label='Training Loss', color='blue')
ax1.set_title('Training Loss Over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Pattern detection events
pattern_epochs = list(range(0, len(losses), 10))[:len(pattern_flags)]
learned_epochs = [epoch for epoch, is_learned in zip(pattern_epochs, pattern_flags) if is_learned]
new_epochs = [epoch for epoch, is_learned in zip(pattern_epochs, pattern_flags) if not is_learned]

ax2.scatter(learned_epochs, [losses[e] for e in learned_epochs], 
           color='red', label='Learned Pattern Detected', s=50, alpha=0.7)
ax2.scatter(new_epochs, [losses[e] for e in new_epochs], 
           color='green', label='New Pattern', s=50, alpha=0.7)
ax2.plot(losses, alpha=0.3, color='gray')
ax2.set_title('Pattern Detection During Training')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Suppression strength over time
ax3.plot(pattern_epochs[:len(suppressions)], suppressions, 'o-', color='orange', label='Suppression Strength')
ax3.set_title('Learning Suppression Strength')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Suppression Strength')
ax3.legend()
ax3.grid(True)

# Learning rate adaptation
ax4.plot(pattern_epochs[:len(lrs)], lrs, 's-', color='purple', label='Adaptive Learning Rate')
ax4.set_title('Adaptive Learning Rate')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('pattern_aware_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_train)
    mse = nn.MSELoss()(predictions, y_train)
    print(f'\nFinal MSE on Training Data: {mse.item():.4f}')

# Statistics
learned_count = sum(pattern_flags)
total_checks = len(pattern_flags)
print(f"Pattern Analysis:")
print(f"- Total pattern checks: {total_checks}")
print(f"- Learned patterns detected: {learned_count}")
print(f"- New patterns: {total_checks - learned_count}")
print(f"- Average suppression strength: {np.mean(suppressions):.3f}")

# Inverse transform predictions for interpretation
predictions = scaler_y.inverse_transform(predictions.numpy())
y_true = scaler_y.inverse_transform(y_train.numpy())
print(f'\nSample Predictions (first 5): {predictions[:5].flatten()}')
print(f'Sample True Values (first 5): {y_true[:5].flatten()}')
