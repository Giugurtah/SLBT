# categorize_data.py

import pandas as pd
import numpy as np
from slbt import Categorizer

print("="*60)
print("DATA CATEGORIZATION FOR SLBT")
print("="*60)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

df = pd.read_csv('df_sla_Tracheostomia2_SpirometriaND.csv', 
                 sep=';',      # Separatore colonne
                 decimal=',',
                 header=0)  # Separatore decimale italiano

print(f"\nOriginal DataFrame shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# ============================================================================
# DEFINE COLUMN TYPES
# ============================================================================

# Columns to categorize (continuous variables)
col_to_cat = [
    'Age_at_onset', 
    'Diagnostic_delay', #si
    'CNS_LS', #si
    'disease_duration', #si
    'rate_of_progression'#si
]

col_to_cat_3  = [
    'MRC_AASS', #si
    'MRC_AAII', #si
    'MRC_Bulbare', #si
    'PUMNS_AASS',  #si
    'PUMNS_AAII',  #si
    'PUMNS_Bulbare',  #si
]

# Columns to drop
col_to_drop = [

]

# Already categorical columns
col_ok = [
    'CUT_OFF_LABILITY_SCALE',  #si
    'Clinical_onset_c', 
    'Spirometry', 
    'FVC', 
    'VENTILATION', 
    'Tracheostomy', 
    'PEG', #si
    'Family_history', 
    'Therapy'
]

# Target variable
targhet_var = ["KINGS_TOT"] #si

# Stratifying variable
stratifing_var = ['Sex']

# ============================================================================
# DROP SPECIFIED COLUMNS
# ============================================================================

df.drop(columns=col_to_drop, axis=1, inplace=True)
print(f"\nAfter dropping columns: {df.shape}")

# ============================================================================
# CHECK AND DROP ROWS WITH NULL VALUES
# ============================================================================

print("\n" + "="*60)
print("CHECKING FOR NULL VALUES")
print("="*60)

null_counts = df.isnull().sum()
total_nulls = null_counts.sum()

if total_nulls > 0:
    print("\nNull values per column:")
    print(null_counts[null_counts > 0].to_string())
    print(f"\nTotal rows with at least one null: {df.isnull().any(axis=1).sum()}")
    
    # Drop rows with nulls
    df_before = len(df)
    df = df.dropna()
    df_after = len(df)
    
    print(f"\n✓ Dropped {df_before - df_after} rows with null values")
    print(f"Final dataset shape: {df.shape}")
else:
    print("\n✓ No null values found!")

if len(df) == 0:
    print("\n⚠️  ERROR: No data left after dropping nulls!")
    exit(1)

# ============================================================================
# CATEGORIZE CONTINUOUS FEATURES
# ============================================================================

print("\n" + "="*60)
print("CATEGORIZING CONTINUOUS FEATURES")
print("="*60)

# Check each column before categorization
valid_cols = []
for col in col_to_cat:
    if col not in df.columns:
        print(f"\n⚠️  {col}: NOT FOUND in DataFrame")
        continue
    
    print(f"\n{col}:")
    print(f"  dtype: {df[col].dtype}")
    print(f"  null count: {df[col].isnull().sum()}")
    print(f"  unique values: {df[col].nunique()}")
    
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"  ⚠️  WARNING: Not numeric! Skipping.")
        continue
    
    # Check if all NaN
    if df[col].isnull().all():
        print(f"  ⚠️  WARNING: All NaN! Skipping.")
        continue
    
    # Check for infinite values
    if np.isinf(df[col]).any():
        print(f"  ⚠️  WARNING: Contains infinite values! Cleaning...")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
    
    # Check for constant values
    if df[col].nunique(dropna=True) <= 1:
        print(f"  ⚠️  WARNING: Constant column! Skipping.")
        continue
    
    print(f"  ✓ Valid for categorization")
    valid_cols.append(col)

print(f"\n{'-'*60}")
print(f"Valid columns: {len(valid_cols)} / {len(col_to_cat)}")
print(f"{'-'*60}")

valid_cols_3 = []
for col in col_to_cat_3:
    if col not in df.columns:
        print(f"\n⚠️  {col}: NOT FOUND in DataFrame")
        continue
    
    print(f"\n{col}:")
    print(f"  dtype: {df[col].dtype}")
    print(f"  null count: {df[col].isnull().sum()}")
    print(f"  unique values: {df[col].nunique()}")
    
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"  ⚠️  WARNING: Not numeric! Skipping.")
        continue
    
    # Check if all NaN
    if df[col].isnull().all():
        print(f"  ⚠️  WARNING: All NaN! Skipping.")
        continue
    
    # Check for infinite values
    if np.isinf(df[col]).any():
        print(f"  ⚠️  WARNING: Contains infinite values! Cleaning...")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
    
    # Check for constant values
    if df[col].nunique(dropna=True) <= 1:
        print(f"  ⚠️  WARNING: Constant column! Skipping.")
        continue
    
    print(f"  ✓ Valid for categorization")
    valid_cols_3.append(col)

if len(valid_cols_3) == 0:
    print("\n⚠️  ERROR: No valid columns to categorize!")
    exit(1)

# Update col_to_cat to only valid columns
col_to_cat = valid_cols
col_to_cat_3 = valid_cols_3

# Categorize each column separately
X_categorical_dict = {}

for col in col_to_cat:
    print(f"\nCategorizing {col}...")
    try:
        cat_single = Categorizer(method='elbow', k_max=8, k_min=2)
        
        # Extract column as numpy array
        col_data = df[col].values
        
        result = cat_single.fit_transform(col_data)
        X_categorical_dict[col] = result.values.ravel()
        
        info = cat_single.get_bin_info('X')
        print(f"  ✓ {cat_single.k_['X']} bins, centers: {np.round(info['centers'], 1)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        # Use original values if categorization fails
        X_categorical_dict[col] = df[col].values

for col in col_to_cat_3:
    print(f"\nCategorizing {col}...")
    try:
        cat_single = Categorizer(method='elbow', k_min=2, k_max=8)
        
        # Extract column as numpy array
        col_data = df[col].values
        
        result = cat_single.fit_transform(col_data)
        X_categorical_dict[col] = result.values.ravel()
        
        info = cat_single.get_bin_info('X')
        print(f"  ✓ {cat_single.k_['X']} bins, centers: {np.round(info['centers'], 1)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        # Use original values if categorization fails
        X_categorical_dict[col] = df[col].values

# Combine into DataFrame
X_categorical = pd.DataFrame(X_categorical_dict, index=df.index)

# ============================================================================
# ADD OTHER COLUMNS
# ============================================================================

print("\n" + "="*60)
print("ADDING OTHER COLUMNS")
print("="*60)

# Add categorical columns (col_ok)
print("\nAdding already categorical columns (col_ok):")
for col in col_ok:
    if col in df.columns:
        X_categorical[col] = df[col].values
        print(f"  ✓ {col} (unique: {df[col].nunique()})")
    else:
        print(f"  ⚠️  {col}: NOT FOUND")

# Add target variable
print("\nAdding target variable:")
for col in targhet_var:
    if col in df.columns:
        X_categorical[col] = df[col].values
        print(f"  ✓ {col} (unique: {df[col].nunique()})")
        print(f"    Distribution: {df[col].value_counts().to_dict()}")
    else:
        print(f"  ⚠️  {col}: NOT FOUND")

# Add stratifying variable
print("\nAdding stratifying variable:")
for col in stratifing_var:
    if col in df.columns:
        X_categorical[col] = df[col].values
        print(f"  ✓ {col} (unique: {df[col].nunique()})")
        print(f"    Distribution: {df[col].value_counts().to_dict()}")
    else:
        print(f"  ⚠️  {col}: NOT FOUND")

# ============================================================================
# FINAL DATASET SUMMARY
# ============================================================================

print("\n" + "="*60)
print("FINAL DATASET SUMMARY")
print("="*60)

print(f"\nFinal dataset shape: {X_categorical.shape}")

print(f"\nColumn types:")
print(f"  - Categorized (from continuous): {len(col_to_cat)}")
print(f"  - Already categorical: {len([c for c in col_ok if c in X_categorical.columns])}")
print(f"  - Target variable: {len([c for c in targhet_var if c in X_categorical.columns])}")
print(f"  - Stratifying variable: {len([c for c in stratifing_var if c in X_categorical.columns])}")
print(f"  - Total: {len(X_categorical.columns)}")

print("\nColumn list:")
print(X_categorical.columns.tolist())

print("\nFirst 10 rows:")
print(X_categorical.head(10))

print("\nData types:")
print(X_categorical.dtypes)

print("\nNull values check:")
final_nulls = X_categorical.isnull().sum().sum()
if final_nulls > 0:
    print(f"  ⚠️  WARNING: {final_nulls} null values remaining!")
    print(X_categorical.isnull().sum()[X_categorical.isnull().sum() > 0])
else:
    print("  ✓ No null values!")

# ============================================================================
# SAVE FILES
# ============================================================================

print("\n" + "="*60)
print("SAVING FILES")
print("="*60)

# Save complete dataset
output_file = 'df_sla3_complete.csv'
X_categorical.to_csv(output_file, index=False)
print(f"✓ Saved complete dataset to: {output_file}")

# ============================================================================
# CREATE SEPARATE VARIABLES
# ============================================================================

print("\n" + "="*60)
print("CREATING SEPARATE VARIABLES FOR SLBT")
print("="*60)

# Features (X)
feature_cols = [c for c in col_to_cat if c in X_categorical.columns]+ \
               [c for c in col_to_cat_3 if c in X_categorical.columns] + \
               [c for c in col_ok if c in X_categorical.columns]
X = X_categorical[feature_cols]
print(f"\nX (features): {X.shape}")
print(f"Columns: {X.columns.tolist()}")

# Target (y)
if targhet_var[0] in X_categorical.columns:
    y = X_categorical[targhet_var[0]]
    print(f"\ny (target): {y.shape}")
    print(f"Unique values: {y.unique()}")
    print(f"Distribution:")
    print(y.value_counts().to_string())
else:
    print(f"\n⚠️  Target variable '{targhet_var[0]}' not found!")
    y = None

# Stratification (x_s)
if stratifing_var[0] in X_categorical.columns:
    x_s = X_categorical[stratifing_var[0]]
    print(f"\nx_s (stratification): {x_s.shape}")
    print(f"Unique values: {x_s.unique()}")
    print(f"Distribution:")
    print(x_s.value_counts().to_string())
else:
    print(f"\n⚠️  Stratifying variable '{stratifing_var[0]}' not found!")
    x_s = None

# Save separate files
X.to_csv('df_sla3_X.csv', index=False)
if y is not None:
    y.to_csv('df_sla3_y.csv', index=False, header=True)
if x_s is not None:
    x_s.to_csv('df_sla3_x_s.csv', index=False, header=True)

print("\n✓ Saved separate files:")
print("  - df_sla3_complete.csv (all columns)")
print("  - df_sla3_X.csv (features only)")
if y is not None:
    print("  - df_sla3_y.csv (target)")
if x_s is not None:
    print("  - df_sla3_x_s.csv (stratification)")

# ============================================================================
# DATA QUALITY CHECK
# ============================================================================

print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

print(f"Final rows: {len(X_categorical)}")
print(f"Null values: {X_categorical.isnull().sum().sum()}")

# ============================================================================
# READY FOR SLBT
# ============================================================================

print("\n" + "="*60)
print("READY FOR SLBT!")
print("="*60)

print("""
You can now train SLBT with:

import pandas as pd
from slbt import SLBT
from slbt.plotting import plot_html

# Load data
X = pd.read_csv('df_sla_X.csv')
y = pd.read_csv('df_sla_y.csv').squeeze()
x_s = pd.read_csv('df_sla_x_s.csv').squeeze()

# Train model
model = SLBT(homogeneity='none', max_depth=5, min_ppi=0.01, min_gpi=0.05)
model.fit(X, y, x_s=x_s)

# Make predictions
predictions = model.predict(X)
accuracy = (predictions == y).sum() / len(y)
print(f"Accuracy: {accuracy:.2%}")

# Visualize tree
plot_html(model, output_file='sla_tree.html', title='SLA Decision Tree')
""")

print("="*60)

