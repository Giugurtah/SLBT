import pandas as pd
import numpy as np
from slbt import SLBT, Categorizer
from slbt.plotting import plot_html


df = pd.read_csv('z_datasets/df_sla_complete_gender.csv', header=0, sep=";")

y = df['KINGS_TOT']
x_s = df['Sex']
X = df.drop(columns=['ID_visit','KINGS_TOT',"ID_pz","Sex"])

homogeneitys = ["none", "A", "B", "AB"]
no_homo = ["AB"]

for homo in no_homo:
    model = SLBT(max_depth=5, homogeneity=homo)
    model.fit(X, y)

    output_file = "lbt_als_gender.html"
    plot_html(
        model,
        output_file=output_file,
        title="LBT ALS"
    )

    model.reporter.results.to_csv("lbt_als_gender.csv")

    numero = input("Inserisci un numero intero: ")
    numero = int(numero)
    model.prune_after_vp(numero)

    output_file = "pruned_lbt_als_gender.html"
    plot_html(
        model,
        output_file=output_file,
        title="LBT ALS PRUNED"
    )

    model.reporter.results.to_csv("pruned_slbt_als_gender.csv")




