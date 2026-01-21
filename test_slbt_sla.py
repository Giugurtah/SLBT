import pandas as pd
import numpy as np
from slbt import SLBT, Categorizer
from slbt.plotting import plot_html

X = pd.read_csv('df_sla_X.csv')

y = pd.read_csv('df_sla_y.csv').squeeze()
x_s = pd.read_csv('df_sla_x_s.csv').squeeze()

model = SLBT(max_depth=5, homogeneity="AB")
model.fit(X, y)

output_file = "non_stratified_tree_vp.html"
plot_html(
    model,
    output_file=output_file,
    title="LBT_SLA_VP",
)



