import pandas as pd
import numpy as np
from slbt import SLBT, Categorizer
from slbt.plotting import plot_html

X = pd.read_csv('df_sla_X.csv')

y = pd.read_csv('df_sla_y.csv').squeeze()
x_s = pd.read_csv('df_sla_x_s.csv').squeeze()

model = SLBT(max_depth=5, homogeneity="AB")
model.fit(X, y, x_s)

output_file = "stratified_tree_AB.html"
plot_html(
    model,
    output_file=output_file,
    title="SLBT_ALS_AB_VP",
)

model.reporter.results.to_csv("stratified_tree_AB.csv")

model.prune_after_vp(10)

output_file = "pruned_stratified_tree_AB.html"
plot_html(
    model,
    output_file=output_file,
    title="LBT_ALS_AB_PRUNED",
)

model.reporter.results.to_csv("pruned_stratified_tree_AB.csv")




