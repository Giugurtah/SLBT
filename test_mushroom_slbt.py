import pandas as pd
import numpy as np
from slbt import SLBT, Categorizer
from slbt.plotting import plot_html


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring',
           'stalk-color-below-ring', 'veil-type', 'veil-color',
           'ring-number', 'ring-type', 'spore-print-color',
           'population', 'habitat']

df = pd.read_csv(url, names=columns)

print(f"Valori mancanti: {df.isnull().sum().sum()}")
df.replace('?', np.nan, inplace=True)
print(f"Valori mancanti dopo conversione '?': {df.isnull().sum().sum()}")
df = df.dropna()
print(f"Righe rimanenti: {len(df)}")

y = df['class']
X = df.drop('class', axis=1)

model = SLBT(max_depth=3, min_impurity= 0, homogeneity="AB")
model.fit(X, y)

output_file = "non_stratified_mushrooms.html"
plot_html(
    model,
    output_file=output_file,
    title="MUSHROOMS"
)
