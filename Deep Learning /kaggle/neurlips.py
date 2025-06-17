import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# --------------------
# Load Data
# --------------------
train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# --------------------
# Feature Extraction (RDKit)
# --------------------
def compute_rdkit_features(smiles):
    """Compute RDKit molecular descriptors from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * 10
        
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.MolMR(mol),
        ]
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return [np.nan] * 10

feature_names = [
    'MolWt', 'LogP', 'TPSA', 'RotBonds', 'HDonors', 'HAcceptors',
    'FracCSP3', 'RingCount', 'HeavyAtoms', 'MolMR'
]

# Compute RDKit features
print("Computing RDKit features for training data...")
train_features = train_df['SMILES'].apply(compute_rdkit_features)
print("Computing RDKit features for test data...")
test_features = test_df['SMILES'].apply(compute_rdkit_features)

train_features_df = pd.DataFrame(train_features.tolist(), columns=feature_names)
test_features_df = pd.DataFrame(test_features.tolist(), columns=feature_names)

# Handle missing values in features using median imputation
imputer = SimpleImputer(strategy='median')
train_features_df = pd.DataFrame(
    imputer.fit_transform(train_features_df), 
    columns=feature_names
)
test_features_df = pd.DataFrame(
    imputer.transform(test_features_df), 
    columns=feature_names
)

# Merge features with original data
train_full = pd.concat([train_df, train_features_df], axis=1)
test_full = pd.concat([test_df, test_features_df], axis=1)

print(f"Training data shape: {train_full.shape}")
print(f"Test data shape: {test_full.shape}")

# --------------------
# Custom Weighted MAE (wMAE)
# --------------------
def compute_wMAE(y_true_dict, y_pred_dict, test_ranges):
    """Compute weighted Mean Absolute Error"""
    weights = []
    errors = []
    
    # Calculate normalization factor
    sqrt_inv_ni = [1 / np.sqrt(len(y_true_dict[prop])) for prop in y_true_dict]
    normalization = sum(sqrt_inv_ni)
    K = len(y_true_dict)
    
    for i, prop in enumerate(y_true_dict):
        ni = len(y_true_dict[prop])
        ri = test_ranges[prop][1] - test_ranges[prop][0]
        
        # Avoid division by zero
        if ri == 0:
            ri = 1.0
            
        wi = (1 / ri) * (K * (1 / np.sqrt(ni)) / normalization)
        mae = mean_absolute_error(y_true_dict[prop], y_pred_dict[prop])
        errors.append(wi * mae)
        
        print(f"{prop}: MAE={mae:.6f}, Weight={wi:.6f}, Weighted_MAE={wi*mae:.6f}")
    
    return np.mean(errors)

# --------------------
# Train XGBoost Models (Updated for XGBoost 2.0+)
# --------------------
target_properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
models = {}
predictions = {}
y_true_dict = {}
y_pred_dict = {}

# Check if CUDA is available
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    cuda_available = result.returncode == 0
except:
    cuda_available = False

device = "cuda" if cuda_available else "cpu"
print(f"Using device: {device}")

for target in target_properties:
    print(f"\nTraining model for {target}...")
    
    # Filter only available target rows
    train_data = train_full[~train_full[target].isna()]
    X_train = train_data[feature_names]
    y_train = train_data[target]
    
    print(f"Training samples for {target}: {len(train_data)}")
    
    # Updated XGBoost parameters for version 2.0+
    model = xgb.XGBRegressor(
        device=device,  # Use 'cuda' for GPU, 'cpu' for CPU
        tree_method='hist',  # 'hist' works for both CPU and GPU
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,  # Use all available cores for CPU
        verbosity=0  # Reduce verbosity to minimize warnings
    )
    
    model.fit(X_train, y_train)
    models[target] = model
    
    # Predict for test set
    predictions[target] = model.predict(test_full[feature_names])
    
    # Store training predictions for wMAE calculation
    y_true_dict[target] = y_train.values
    y_pred_dict[target] = model.predict(X_train)

# --------------------
# Compute Property Ranges for wMAE
# --------------------
test_ranges = {}
for prop in target_properties:
    prop_values = train_full[prop].dropna()
    test_ranges[prop] = (prop_values.min(), prop_values.max())
    print(f"{prop} range: [{test_ranges[prop][0]:.4f}, {test_ranges[prop][1]:.4f}]")

# --------------------
# Evaluate
# --------------------
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

wmae_score = compute_wMAE(y_true_dict, y_pred_dict, test_ranges)
print(f"\n✅ Weighted MAE (wMAE): {wmae_score:.6f}")

# Individual property performance
print("\nIndividual Property Performance:")
for target in target_properties:
    mae = mean_absolute_error(y_true_dict[target], y_pred_dict[target])
    print(f"{target}: MAE = {mae:.6f}")

# --------------------
# Create Submission
# --------------------
submission_df = test_df[['id']].copy()
for prop in target_properties:
    submission_df[prop] = predictions[prop]

# Check for any NaN predictions
for prop in target_properties:
    nan_count = submission_df[prop].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN predictions for {prop}")
        # Fill NaN with median of training data
        median_val = train_full[prop].median()
        submission_df[prop].fillna(median_val, inplace=True)
        print(f"Filled NaN values with median: {median_val:.6f}")

submission_df.to_csv("submission.csv", index=False)
print(f"\n📄 Submission saved as 'submission.csv'")
print(f"Submission shape: {submission_df.shape}")

# Display first few predictions
print("\nFirst 5 predictions:")
print(submission_df.head())
