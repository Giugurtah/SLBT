# example_3_slbt_stratified.py
# example_stratified_simple.py

import pandas as pd
import numpy as np
from slbt import SLBT, Categorizer
from slbt.plotting import plot_html

print("="*60)
print("STRATIFIED SLBT - Simple Example")
print("="*60)

# ============================================================================
# STEP 1: Generate stratified data
# ============================================================================
print("\nStep 1: Generating gender-stratified data...")

np.random.seed(42)
n_samples = 600

# Generate gender (stratification variable)
gender = np.array(['male'] * (n_samples//2) + ['female'] * (n_samples//2))
np.random.shuffle(gender)

ages = []
incomes = []
credit_scores = []
approvals = []

# Generate data with different patterns per gender
for g in ['male', 'female']:
    mask = gender == g
    n = mask.sum()
    
    if g == 'male':
        # For males: Income matters more
        age = np.random.normal(45, 12, n)
        income = np.concatenate([
            np.random.normal(35000, 5000, n//3),   # Low income -> rejected
            np.random.normal(60000, 8000, n//3),   # Medium income -> mixed
            np.random.normal(90000, 10000, n//3 + n%3),  # High income -> approved
        ])
        credit = np.random.normal(650, 80, n)
        
        # Approval based mainly on income
        approval = ['rejected' if inc < 50000 else 'approved' for inc in income]
        
    else:  # female
        # For females: Credit score matters more
        age = np.random.normal(42, 10, n)
        income = np.random.normal(55000, 15000, n)
        credit = np.concatenate([
            np.random.normal(580, 40, n//3),   # Low credit -> rejected
            np.random.normal(680, 50, n//3),   # Medium credit -> mixed
            np.random.normal(770, 40, n//3 + n%3),  # High credit -> approved
        ])
        
        # Approval based mainly on credit
        approval = ['rejected' if cred < 640 else 'approved' for cred in credit]
    
    ages.append(age)
    incomes.append(income)
    credit_scores.append(credit)
    approvals.extend(approval)

# Combine
ages = np.concatenate(ages)
incomes = np.concatenate(incomes)
credit_scores = np.concatenate(credit_scores)

# Create DataFrame
df = pd.DataFrame({
    'age': ages,
    'income': incomes,
    'credit_score': credit_scores,
    'gender': gender
})

y = pd.Series(approvals)

print(f"  Total samples: {len(df)}")
print(f"  Males: {(df['gender'] == 'male').sum()}")
print(f"  Females: {(df['gender'] == 'female').sum()}")

print("\nData by gender:")
for g in ['male', 'female']:
    mask = df['gender'] == g
    print(f"\n  {g.upper()}:")
    print(f"    Avg Income: ${df.loc[mask, 'income'].mean():,.0f}")
    print(f"    Avg Credit: {df.loc[mask, 'credit_score'].mean():.0f}")
    print(f"    Approval Rate: {(y[mask] == 'approved').sum() / mask.sum() * 100:.1f}%")

print("\nFirst 10 rows:")
print(df.head(10))

# ============================================================================
# STEP 2: Categorize continuous variables
# ============================================================================
print("\n" + "="*60)
print("Step 2: Categorizing continuous features...")
print("="*60)

numeric_cols = ['age', 'income', 'credit_score']
cat = Categorizer(method='elbow', k_max=4, k_min=2)
X_categorical = cat.fit_transform(df[numeric_cols])

print("\nBins found:")
for col in numeric_cols:
    info = cat.get_bin_info(col)
    print(f"  {col}: {cat.k_[col]} bins")
    print(f"    Centers: {np.round(info['centers'], 0)}")

print("\nCategorized data (first 10):")
print(X_categorical.head(10))

# ============================================================================
# STEP 3: Train SLBT with stratification
# ============================================================================
print("\n" + "="*60)
print("Step 3: Training SLBT with gender stratification...")
print("="*60)

model = SLBT(
    homogeneity='none',  # Allow different patterns per stratum
    max_depth=4,
    min_ppi=0,
    min_gpi=0,
    min_impurity=0
)

# Fit with gender as stratification variable
print("\nFitting model with x_s=gender...")
model.fit(X_categorical, y, x_s=df['gender'])

print("✓ Model trained successfully")

# ============================================================================
# STEP 5: Generate tree visualization
# ============================================================================
print("\n" + "="*60)
print("Step 5: Generating tree visualization...")
print("="*60)

output_file = "stratified_tree_None.html"
plot_html(
    model,
    output_file=output_file,
    title="SLBT Stratified Tree - "
)

print(f"✓ Tree visualization saved to: {output_file}")

# ============================================================================
# STEP 4: Make predictions
# ============================================================================
print("\n" + "="*60)
print("Step 4: Making predictions...")
print("="*60)

predictions = model.predict(X_categorical)

# Overall accuracy
accuracy = (predictions == y).sum() / len(y)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Accuracy by gender
for g in ['male', 'female']:
    mask = df['gender'] == g
    acc_g = (predictions[mask] == y[mask]).sum() / mask.sum()
    print(f"  {g.capitalize()} Accuracy: {acc_g:.2%}")

# Confusion matrix
from collections import Counter
print("\nPrediction distribution:")
pred_dist = Counter(predictions)
for label, count in pred_dist.items():
    print(f"  {label}: {count} ({count/len(predictions)*100:.1f}%)")

print("\nActual distribution:")
actual_dist = Counter(y)
for label, count in actual_dist.items():
    print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")

# Show examples
print("\nSample predictions:")
sample_idx = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
sample_df = pd.DataFrame({
    'gender': df.loc[sample_idx, 'gender'].values,
    'age': df.loc[sample_idx, 'age'].round(0).astype(int).values,
    'income': df.loc[sample_idx, 'income'].round(-3).astype(int).values,
    'credit': df.loc[sample_idx, 'credit_score'].round(0).astype(int).values,
    'actual': y.iloc[sample_idx].values,
    'predicted': [predictions[i] for i in sample_idx],
    'correct': ['✓' if predictions[i] == y.iloc[i] else '✗' for i in sample_idx]
})
print(sample_df.to_string(index=False))


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"""
Dataset:
  - Total samples: {len(df)}
  - Males: {(df['gender'] == 'male').sum()}
  - Females: {(df['gender'] == 'female').sum()}
  - Approval rate: {(y == 'approved').sum() / len(y) * 100:.1f}%

Model Performance:
  - Overall Accuracy: {accuracy:.2%}
  - Male Accuracy: {(predictions[df['gender'] == 'male'] == y[df['gender'] == 'male']).sum() / (df['gender'] == 'male').sum():.2%}
  - Female Accuracy: {(predictions[df['gender'] == 'female'] == y[df['gender'] == 'female']).sum() / (df['gender'] == 'female').sum():.2%}

Visualization:
  - File: {output_file}
  - Open with: open {output_file}

Key Feature:
  The tree learns DIFFERENT patterns for males vs females:
  - Males: Income-driven decisions
  - Females: Credit-score-driven decisions
""")

print("="*60)
print("To view the interactive tree, run:")
print(f"  open {output_file}")
print("="*60)