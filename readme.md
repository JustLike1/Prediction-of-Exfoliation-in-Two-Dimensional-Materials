# 2D Material Exfoliability Prediction System

## 📋 Project Overview

This project uses 38 features to characterize the layered structure of 2D materials, including:
- **7 Ratio Features**: Interlayer/intralayer structural feature ratios
- **21 Density Distribution Features**: Atomic density waveform features along the z-direction
- **7 Structural Features**: Topological features of the original crystal structure
- **3 Global Features**: Periodicity, symmetry, and layer consistency of the crystal

## 🚀 Quick Start

### Requirements

```bash
# Main Dependencies
- Python 3.10+
- XGBoost
- PyTorch & PyTorch Geometric
- Pymatgen
- ASE (Atomic Simulation Environment)
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, SHAP
```

### Workflow

#### Step 1: Crystal Structure Layering

Use `no_sys.ipynb` to decompose crystal structures into intralayer and interlayer components.

**Configuration:**
- Modify `input_folder` and `output_folder` paths in Cells 5 and 6
- Input: Original VASP structure files
- Output: Separated max (interlayer) and min (intralayer) structure files

**Optional Tool:**
- `no_sys_draw.ipynb`: Visualize atomic density distribution maps

---

#### Step 2: Feature Extraction

Extract features from crystal structures using Python scripts.

**Training Dataset Generation:**
```python
# Use processData_38_features.py
# Modify the following paths:
fail_dir = "data/split/fail"          # Negative samples (non-exfoliable structures)
success_dir = "data/split/success"    # Positive samples (exfoliable structures)
vasp_fail_dir = "data/vaspFail"       # Original negative sample VASP files
vasp_success_dir = "data/vaspSuccess" # Original positive sample VASP files
```

**Prediction Dataset Generation:**
```python
# Use processData_unlabeled_38_features.py
# Generate features for unlabeled structures
```

**Output:** CSV file containing 38 features and labels (training set) or features only (prediction set)

---

#### Step 3: Model Training

Train the model and perform cross-validation using `XGBoost_train.ipynb`.

**Configuration:**
```python
# Modify data path
csv_path = 'data/dataset/features_final_38_new.csv'

# Training parameters
SPLIT_SEED = 43          # Random seed for data splitting
XGBOOST_MODEL_SEED = 51  # Random seed for model training
N_SPLITS = 5             # Number of cross-validation folds
```

**Training Outputs:**
- `training_results/` folder contains:
  - `complete_results.json`: Complete training results
  - `average_metrics.json`: Average performance metrics
  - `feature_importance.json`: Feature importance rankings
  - `confusion_matrices.png`: Confusion matrix visualization
  - `roc_curves.png`: ROC curves
  - `learning_curves.png`: Learning curves
  - `shap_summary_top10.png`: SHAP feature explanations (Top 10)
  - CSV files: Detailed learning curves and ROC data for each fold

---

#### Step 4: Structure Prediction

Predict new structures using `XGBoost_predict.ipynb`.

**Configuration:**
```python
# Training data path (for training the final model)
TRAIN_CSV = 'data/dataset/features_final_38.csv'

# Unlabeled data path for prediction
UNLABELED_CSV = 'data/dataset/features_38_unlabeled.csv'

# Model and results save paths
MODEL_PATH = 'models/xgboost_full_model.json'
OUTPUT_CSV = 'results/predictions.csv'
```

**Prediction Outputs:**
- Saved model file (`.json` format)
- Prediction results CSV file containing:
  - `filename`: Structure filename
  - `predicted_label`: Predicted label (0=non-exfoliable, 1=exfoliable)
  - `probability`: Exfoliability probability

## 📊 Feature Description

### Ratio Features (7)
Feature ratios extracted from max/min structure pairs, reflecting interlayer vs. intralayer differences:
- Node electronegativity ratio, atomic radius ratio
- Edge distance ratio, bond type ratio, edge weight ratio
- Edge count ratio, edge-to-node ratio

### Density Distribution Features (21)

**Basic Statistics (4)**
- Maximum density, minimum density, density ratio, density difference

**Peak-Valley Features (2)**
- Peak range, peak distance standard deviation

**Morphological Features (6)**
- Skewness, kurtosis, standard deviation, range, energy, entropy

**Derivative Features (5)**
- First derivative statistics (mean, std, max)
- Second derivative statistics (mean, std)

**Frequency Domain Features (2)**
- Maximum frequency amplitude, mean frequency amplitude

**Periodicity Features (2)**
- Periodicity lag, periodicity strength

### Structural Features (7)
Topological features from original VASP files:
- Average electronegativity, average atomic radius
- Average edge distance, average bond type, average edge weight
- Edge count, edge-to-node ratio

### Global Features (3)
Global structural properties of the crystal:
- Layer atomic composition consistency
- Periodicity strength along z-axis
- Symmetry along z-axis

## 📈 Model Performance

Training results are saved in the `training_results/` folder, including:
- Accuracy, AUC, F1 score, and other metrics
- Precision and recall for each class
- Feature importance analysis
- SHAP value explanations

## 🔧 Custom Configuration

### Modify Model Parameters

Adjust hyperparameters in `XGBoost_train.ipynb`:
```python
params = {
    'max_depth': 9,
    'learning_rate': 0.042,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'scale_pos_weight': 2.23,
    'min_child_weight': 1,
    'gamma': 0.2
}
```

### Modify Feature Extraction Parameters

Adjust in `processData_38_features.py`:
```python
# Layer separation tolerance
z_tolerance = 0.5

# Edge connection threshold
cutoff_factor = 1.2

# Periodicity detection range
max_period = 5
```
