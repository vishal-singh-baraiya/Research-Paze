import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Generate synthetic data
N = 500
x1 = np.random.normal(0, 10, N)  # Distance from school
x2 = np.random.normal(0, 10, N)  # Area
X = np.vstack((x1, x2)).T

# Create target variables with noise
price = x1 + 1.3 * x2 + np.random.normal(0, 5, N)
rent = 0.8 * x1 + 1.5 * x2 + np.random.normal(0, 3, N)
y = np.vstack((price, rent)).T

# Standardize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

class Cell:
    def __init__(self, task_type, input_dim=2, hidden_dim=8):
        """Initialize a cell with specific task and properties"""
        self.task = task_type  # 'price' or 'rent'
        self.position = torch.randn(input_dim, requires_grad=False)
        self.age = 0
        self.energy = 100.0
        self.generation = 0
        self.specialization = 0.0  # 0.0 = generalist, 1.0 = specialist
        self.error_history = []
        self.memory = []
        self.id = random.randint(10000, 99999)  # Unique cell identifier
        self.last_trained = 0  # Track when the cell was last trained
        
        # Fixed architecture to start
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize optimizer with lower learning rate
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.005)
        
    def predict(self, x):
        """Forward pass through the neural network"""
        return self.net(x)
    
    def train(self, x, y_true, epoch):
        """Train the cell on a single example"""
        # Only train every few epochs based on age
        if epoch % max(1, int(self.age / 10)) != 0:
            return 0.0  # Skip training this epoch
            
        self.last_trained = epoch
            
        # Standard training
        self.optimizer.zero_grad()
        y_pred = self.predict(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        
        # Experience replay (from memory)
        if len(self.memory) > 5 and random.random() < 0.3:
            mem_idx = random.randint(0, len(self.memory)-1)
            mem_x, mem_y = self.memory[mem_idx]
            mem_pred = self.predict(mem_x)
            memory_loss = nn.functional.mse_loss(mem_pred, mem_y)
            loss = 0.7 * loss + 0.3 * memory_loss
        
        loss.backward()
        
        # Gradient clipping to prevent drastic changes
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        
        self.optimizer.step()
        self.error_history.append(loss.item())
        
        # Selective memory (store only difficult examples)
        if loss.item() > 0.1:
            self.memory.append((x.detach().cpu(), y_true.detach().cpu()))
            if len(self.memory) > 20:  # Limit memory size
                self.memory.pop(0)
                
        # Update cell state
        self.age += 1
        # Very minimal energy cost - only for training
        self.energy -= loss.item() * 0.5  # Much lower energy cost
        self.energy = max(0, min(self.energy, 150))  # Cap energy between 0 and 150
        
        return loss.item()
    
    def attention(self, other_cells):
        """Maintain spatial diversity by moving away from crowded areas"""
        if not other_cells:
            return
            
        # Calculate distances to all other cells
        others = torch.stack([c.position for c in other_cells])
        dists = torch.norm(others - self.position, dim=1)
        
        # Find cells that are too close
        close_idxs = (dists < 1.5).nonzero().flatten()
        
        if len(close_idxs) > 0:
            # Move away from crowded areas
            self.position = self.position + 0.1 * torch.randn_like(self.position)
            # No energy cost for movement
    
    def average_error(self, window=10):
        """Calculate recent average error"""
        if not self.error_history:
            return 1.0  # Default high error if no history
        if len(self.error_history) < window:
            return np.mean(self.error_history)
        return np.mean(self.error_history[-window:])
    
    def mutate(self, mutation_rate=0.05):
        """Apply random mutations to neural network weights"""
        with torch.no_grad():
            for param in self.net.parameters():
                if random.random() < mutation_rate:
                    param.data += torch.randn_like(param) * 0.05  # Smaller mutations
        
        # No energy cost for mutation
        return self
    
    # Add energy periodically
    def metabolize(self, epoch):
        """Cell gains energy over time"""
        # Add energy every 5 epochs
        if epoch % 5 == 0:
            self.energy += 10
            self.energy = min(150, self.energy)  # Cap at maximum

def evolve_cells(cells, epoch, max_cells=20, min_cells_per_task=3):
    """Apply evolutionary pressures to the cell population"""
    # First, let cells metabolize (gain energy)
    for cell in cells:
        cell.metabolize(epoch)
    
    new_cells = []
    task_counts = {'price': 0, 'rent': 0}
    
    # Count cells by task
    for cell in cells:
        task_counts[cell.task] += 1
    
    for cell in cells:
        # Very minimal base energy cost
        cell.energy -= 0.1  # Tiny energy cost per epoch
        
        # Death conditions - much more lenient
        if cell.energy <= 0:
            continue  # Cell dies from energy depletion
        if cell.age > 100 and cell.average_error() > 0.5 and epoch - cell.last_trained > 10:
            continue  # Cell dies only if old, bad performance, and not recently trained
            
        # Cell survives
        new_cells.append(cell)
        
        # Reproduction conditions - much more controlled
        can_reproduce = (
            cell.age > 20 and  # Older reproduction age
            cell.average_error() < 0.3 and  # Reasonable error tolerance
            cell.energy > 80 and  # Higher energy requirement
            len(cells) < max_cells and
            task_counts[cell.task] < max_cells//2 and
            random.random() < 0.1  # Only 10% chance per eligible cell
        )
        
        if can_reproduce:
            # Create child with inheritance and mutation
            child = Cell(task_type=cell.task)
            child.net.load_state_dict(cell.net.state_dict())
            
            # Position inheritance with mutation
            child.position = cell.position + 0.5 * torch.randn_like(cell.position)
            
            # Genetic inheritance
            child.generation = cell.generation + 1
            child.specialization = min(0.9, cell.specialization + 0.1)  # Increased specialization
            
            # Energy transfer from parent to child
            cell.energy -= 30  # Energy cost to parent
            child.energy = 70  # Good starting energy for child
            
            # Apply mutations
            if random.random() < 0.5:  # 50% chance of mutation
                child.mutate()
                
            new_cells.append(child)
            task_counts[cell.task] += 1
    
    # Ensure minimum cells per task - but only add if below threshold
    for task in task_counts:
        while task_counts[task] < min_cells_per_task:
            new_cells.append(Cell(task_type=task))
            task_counts[task] += 1
            
    return new_cells

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred)**2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to prevent division by zero
    
    # Explained Variance Score
    var_y = np.var(y_true)
    explained_var = 1 - (np.var(y_true - y_pred) / (var_y + 1e-10))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_var
    }

def safe_mean(array):
    """Calculate mean safely, avoiding empty array issues"""
    if not array or len(array) == 0:
        return 0.0
    return np.mean(array)

def evaluate_model(cells, X_data, y_data, scaler_y=None):
    """Evaluate the cell ecosystem with comprehensive metrics"""
    with torch.no_grad():
        price_cells = [c for c in cells if c.task == 'price']
        rent_cells = [c for c in cells if c.task == 'rent']
        
        # Handle case with no cells
        if not price_cells or not rent_cells:
            return {
                'price': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0, 'r2': 0.0, 'explained_variance': 0.0},
                'rent': {'mse': 1.0, 'rmse': 1.0, 'mae': 1.0, 'r2': 0.0, 'explained_variance': 0.0},
                'avg_mse': 1.0
            }
        
        # Predict price
        price_preds = []
        for x in X_data:
            x = x.unsqueeze(0)
            cell_preds = [cell.predict(x).item() for cell in price_cells]
            price_preds.append(np.mean(cell_preds))
            
        # Predict rent
        rent_preds = []
        for x in X_data:
            x = x.unsqueeze(0)
            cell_preds = [cell.predict(x).item() for cell in rent_cells]
            rent_preds.append(np.mean(cell_preds))
            
        # Convert tensors to numpy for calculation
        y_data_np = y_data.cpu().numpy()
        
        # Calculate metrics
        price_metrics = calculate_metrics(y_data_np[:, 0], np.array(price_preds))
        rent_metrics = calculate_metrics(y_data_np[:, 1], np.array(rent_preds))
        
        # If scaler was provided, also calculate metrics in original scale
        if scaler_y is not None:
            # Create arrays in the shape expected by the scaler
            pred_scaled = np.zeros_like(y_data_np)
            pred_scaled[:, 0] = price_preds
            pred_scaled[:, 1] = rent_preds
            
            # Inverse transform to original scale
            y_true_orig = scaler_y.inverse_transform(y_data_np)
            y_pred_orig = scaler_y.inverse_transform(pred_scaled)
            
            # Calculate metrics in original scale
            price_metrics_orig = calculate_metrics(y_true_orig[:, 0], y_pred_orig[:, 0])
            rent_metrics_orig = calculate_metrics(y_true_orig[:, 1], y_pred_orig[:, 1])
            
            # Add these to the results with a prefix
            for k, v in price_metrics_orig.items():
                price_metrics[f'orig_{k}'] = v
            for k, v in rent_metrics_orig.items():
                rent_metrics[f'orig_{k}'] = v
        
        return {
            'price': price_metrics,
            'rent': rent_metrics,
            'avg_mse': (price_metrics['mse'] + rent_metrics['mse']) / 2
        }

def assess_fit(train_history, val_history):
    """
    Analyze learning curves to determine if model is overfitting, underfitting, or well-fitted
    """
    results = {}
    
    # Get the final values for each metric
    train_final = train_history[-1]
    val_final = val_history[-1]
    
    # Check MSE
    train_mse = train_final['avg_mse']
    val_mse = val_final['avg_mse']
    mse_gap = val_mse - train_mse
    
    if train_mse > 0.15:  # High training error
        results['mse'] = "underfitting"
    elif mse_gap > 0.1:  # Big gap between training and validation
        results['mse'] = "overfitting"
    else:
        results['mse'] = "well-fitted"
    
    # Check R² for price prediction
    train_r2_price = train_final['price']['r2']
    val_r2_price = val_final['price']['r2']
    r2_gap_price = train_r2_price - val_r2_price
    
    if train_r2_price < 0.7:  # Low R² on training
        results['r2_price'] = "underfitting"
    elif r2_gap_price > 0.2:  # Big gap between training and validation
        results['r2_price'] = "overfitting"
    else:
        results['r2_price'] = "well-fitted"
    
    # Check R² for rent prediction
    train_r2_rent = train_final['rent']['r2']
    val_r2_rent = val_final['rent']['r2']
    r2_gap_rent = train_r2_rent - val_r2_rent
    
    if train_r2_rent < 0.7:  # Low R² on training
        results['r2_rent'] = "underfitting"
    elif r2_gap_rent > 0.2:  # Big gap between training and validation
        results['r2_rent'] = "overfitting"
    else:
        results['r2_rent'] = "well-fitted"
    
    # Overall assessment
    if "overfitting" in results.values():
        results['overall'] = "overfitting"
    elif "underfitting" in results.values():
        results['overall'] = "underfitting"
    else:
        results['overall'] = "well-fitted"
    
    return results

def train_ecosystem(X_tensor, y_tensor, epochs=50, test_split=0.2):
    """Train the entire cell ecosystem with comprehensive evaluation"""
    # Split data into train/validation/test
    val_size = int(X_tensor.size(0) * 0.1)  # 10% for validation
    test_size = int(X_tensor.size(0) * test_split)  # 20% for test
    train_size = X_tensor.size(0) - val_size - test_size  # Rest for training
    
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    
    X_val = X_tensor[train_size:train_size+val_size]
    y_val = y_tensor[train_size:train_size+val_size]
    
    X_test = X_tensor[train_size+val_size:]
    y_test = y_tensor[train_size+val_size:]
    
    # Initialize cells
    cells = [Cell('price') for _ in range(5)] + [Cell('rent') for _ in range(5)]
    
    # Training history
    history = {
        'cells_count': [], 
        'price_cells': [],
        'rent_cells': [],
        'train_metrics': [],
        'val_metrics': [],
        'test_metrics': []
    }
    
    for epoch in range(epochs):
        epoch_errors = []
        
        # Process data in mini-batches
        batch_size = 32
        permutation = torch.randperm(train_size)
        
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i+batch_size]
            x_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # Each cell processes the batch
            for cell in cells:
                target_idx = 0 if cell.task == 'price' else 1
                batch_errors = []
                
                # Process each example in batch
                for j in range(min(10, x_batch.size(0))):  # Limit to 10 examples per batch
                    xi = x_batch[j:j+1]
                    yi = y_batch[j:j+1, target_idx:target_idx+1]
                    
                    # Train cell on example
                    error = cell.train(xi, yi, epoch)
                    if error > 0:  # Only add non-zero errors (when training occurred)
                        batch_errors.append(error)
                
                # After batch, update cell relationships
                cell.attention([c for c in cells if c != cell])
                
                # Only add non-NaN errors
                valid_errors = [e for e in batch_errors if not np.isnan(e)]
                if valid_errors:
                    epoch_errors.extend(valid_errors)
                
        # After epoch, evolve the ecosystem
        cells = evolve_cells(cells, epoch)
        
        # Record population metrics
        price_cells = len([c for c in cells if c.task == 'price'])
        rent_cells = len([c for c in cells if c.task == 'rent'])
        
        history['cells_count'].append(len(cells))
        history['price_cells'].append(price_cells)
        history['rent_cells'].append(rent_cells)
        
        # Evaluate on all datasets
        train_metrics = evaluate_model(cells, X_train, y_train, scaler_y)
        val_metrics = evaluate_model(cells, X_val, y_val, scaler_y)
        
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Evaluate on test set only occasionally
        if epoch % 10 == 0 or epoch == epochs-1:
            test_metrics = evaluate_model(cells, X_test, y_test, scaler_y)
            history['test_metrics'].append(test_metrics)
            
        # Print progress with more metrics
        if epoch % 5 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1}/{epochs} | " 
                  f"Cells: {len(cells)} (P:{price_cells}/R:{rent_cells}) | "
                  f"Train MSE: {train_metrics['avg_mse']:.4f} | "
                  f"Val MSE: {val_metrics['avg_mse']:.4f} | "
                  f"Train R² (Price/Rent): {train_metrics['price']['r2']:.2f}/{train_metrics['rent']['r2']:.2f} | "
                  f"Val R² (Price/Rent): {val_metrics['price']['r2']:.2f}/{val_metrics['rent']['r2']:.2f}")
    
    # Final evaluation
    test_metrics = evaluate_model(cells, X_test, y_test, scaler_y)
    
    # Assess overfitting/underfitting
    fit_assessment = assess_fit(history['train_metrics'], history['val_metrics'])
    
    # Print final assessment
    print("\nModel Fit Assessment:")
    for metric, status in fit_assessment.items():
        print(f"  {metric}: {status}")
    
    print("\nFinal Test Metrics:")
    print(f"  MSE: {test_metrics['avg_mse']:.4f}")
    print(f"  R² (Price): {test_metrics['price']['r2']:.4f}")
    print(f"  R² (Rent): {test_metrics['rent']['r2']:.4f}")
    print(f"  RMSE (Price): {test_metrics['price']['rmse']:.4f}")
    print(f"  RMSE (Rent): {test_metrics['rent']['rmse']:.4f}")
    
    # In original units (if scaler was provided)
    if 'orig_rmse' in test_metrics['price']:
        print("\nMetrics in Original Units:")
        print(f"  RMSE (Price): ${test_metrics['price']['orig_rmse']:.2f}")
        print(f"  RMSE (Rent): ${test_metrics['rent']['orig_rmse']:.2f}")
        print(f"  MAE (Price): ${test_metrics['price']['orig_mae']:.2f}")
        print(f"  MAE (Rent): ${test_metrics['rent']['orig_mae']:.2f}")
    
    # Create a simple accuracy metric for classification-like evaluation
    # Accuracy = % of predictions within 10% of true value
    y_test_np = y_test.cpu().numpy()
    price_preds = []
    rent_preds = []
    
    for x in X_test:
        x = x.unsqueeze(0)
        # Price predictions
        cell_preds = [cell.predict(x).item() for cell in cells if cell.task == 'price']
        price_preds.append(np.mean(cell_preds) if cell_preds else 0)
        
        # Rent predictions
        cell_preds = [cell.predict(x).item() for cell in cells if cell.task == 'rent']
        rent_preds.append(np.mean(cell_preds) if cell_preds else 0)
    
    # Calculate "accuracy" as percentage of predictions within 10% of true value
    price_accuracy = np.mean(np.abs(np.array(price_preds) - y_test_np[:, 0]) < 0.1 * np.abs(y_test_np[:, 0]))
    rent_accuracy = np.mean(np.abs(np.array(rent_preds) - y_test_np[:, 1]) < 0.1 * np.abs(y_test_np[:, 1]))
    
    print("\nAccuracy (% predictions within 10% of true value):")
    print(f"  Price Accuracy: {price_accuracy:.2%}")
    print(f"  Rent Accuracy: {rent_accuracy:.2%}")
    
    return cells, history, fit_assessment

# Run the training
print("Starting Cell Ecosystem Training...")
final_cells, training_history, fit_assessment = train_ecosystem(X_tensor, y_tensor, epochs=50)
print("Training complete!")

# Print summary statistics
price_cells = len([c for c in final_cells if c.task == 'price'])
rent_cells = len([c for c in final_cells if c.task == 'rent'])
generations = [c.generation for c in final_cells]
specializations = [c.specialization for c in final_cells]

print(f"\nFinal Ecosystem Statistics:")
print(f"Total Cells: {len(final_cells)}")
print(f"Price Cells: {price_cells}")
print(f"Rent Cells: {rent_cells}")
print(f"Max Generation: {max(generations) if generations else 0}")
print(f"Avg Specialization: {np.mean(specializations):.2f}")

# Print a matrix of metrics for the last epoch
print("\nMetrics Matrix for Final Epoch:")
print("-" * 80)
print(f"{'Metric':<20} | {'Training':<20} | {'Validation':<20} | {'Test':<20}")
print("-" * 80)

train_metrics = training_history['train_metrics'][-1]
val_metrics = training_history['val_metrics'][-1]
test_metrics = evaluate_model(final_cells, X_tensor[int(X_tensor.size(0)*0.8):], y_tensor[int(X_tensor.size(0)*0.8):], scaler_y)

metrics_to_show = ['mse', 'rmse', 'r2', 'mae']
tasks = ['price', 'rent']

for task in tasks:
    print(f"{task.upper()}")
    for metric in metrics_to_show:
        train_val = train_metrics[task][metric]
        val_val = val_metrics[task][metric]
        test_val = test_metrics[task][metric]
        print(f"{metric.upper():<20} | {train_val:<20.4f} | {val_val:<20.4f} | {test_val:<20.4f}")
    print("-" * 80)

# Determine if the model is underfitting, overfitting, or well-fitted
print("\nFit Assessment Summary:")
if fit_assessment['overall'] == 'underfitting':
    print("The model is UNDERFITTING - consider increasing model capacity or training longer")
elif fit_assessment['overall'] == 'overfitting':
    print("The model is OVERFITTING - consider adding regularization or reducing model complexity")
else:
    print("The model is WELL-FITTED - good balance between bias and variance")
