import numpy as np
import pandas as pd

# Generate dataset
np.random.seed(42)
n_samples = 100

# Generate X (features) - 3 categorical features
X = pd.DataFrame({
    'feature1': np.random.choice(['A', 'B', 'C'], n_samples),
    'feature2': np.random.choice(['X', 'Y'], n_samples),
    'feature3': np.random.choice(['Low', 'High'], n_samples)
})

# Generate y (target) - 2 classes
y = pd.Series(np.random.choice(['Class0', 'Class1'], n_samples))

# Generate x_s (stratification variable) - 2 strata
x_s = pd.Series(np.random.choice(['Stratum1', 'Stratum2'], n_samples))


# Test _stratified_contingency
from slbt import SLBT, Categorizer
model = SLBT(max_depth=3, homogeneity="none")
model.fit(X, y, x_s)

output_file = "test.html"
plot_html(
    model,
    output_file=output_file,
    title="LBT_SLA_AB_6"
)



