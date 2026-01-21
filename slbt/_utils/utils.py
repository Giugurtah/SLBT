import pandas as pd
import numpy as np


#*Function to build the contingency matrix between two variables
def _contingency_matrix(x, y, norm=False):
    if(norm):
        F_final = pd.crosstab(index = x, columns = y, normalize = True).fillna(0)
    else:
        F = pd.crosstab(x, y, dropna=False)
        F_final = F.div(F.sum(axis=1), axis=0).fillna(0)
    

    return F_final.to_numpy(dtype=np.float64)

#*Function to build the list of contingency matrices stratified by a third variable
def _stratified_contingency(x, y, x_s, norm=False):
    # Sizes and levels
    x_levels = sorted(np.unique(x))
    y_levels = sorted(np.unique(y))
    s_levels = sorted(np.unique(x_s))

    I = len(x_levels)
    J = len(y_levels)
    K = len(s_levels)

    # Conditioned matrix
    F_total = pd.crosstab(x, y, dropna=False)
    F_total_cond = F_total.div(F_total.sum(axis=1), axis=0).fillna(0)
    
    F_total_cond_np = F_total_cond.to_numpy()

    # Inizialize F
    F = np.zeros((K, I, J), dtype=np.float64)
    
    for k, s in enumerate(s_levels):
        mask = (x_s == s)

        ct = pd.crosstab(
            pd.Categorical(x[mask], categories=x_levels),
            pd.Categorical(y[mask], categories=y_levels),
            dropna=False
        )

        ct = ct.reindex(index=x_levels, columns=y_levels, fill_value=0)

        F[k, :, :] = ct.values

    N_total = F.sum()
    F = F / N_total

    if(norm is False):
        return F
    
    # Inizialize Fs
    Fs = np.zeros((K, I, J), dtype=np.float64)

    for i in range(I):
        for j in range(J):
            # Somma delle frequenze assolute di (X=i, Y=j) su tutti gli strati
            tot = 0
            for k in range(K):
                tot += F[k, i, j]  # FIX: era "tot = F[k,i,j]" senza +=
            
            # Distribuisci la probabilità condizionata proporzionalmente
            if tot > 0:
                for k in range(K):
                    # Fs[k,i,j] = P(Y=j|X=i) * peso_strato_k
                    # dove peso_strato_k = F[k,i,j] / sum_k(F[k,i,j])
                    Fs[k, i, j] = F_total_cond_np[i, j] * (F[k, i, j] / tot)  # FIX: accesso corretto
            else:
                # Se non ci sono campioni, distribuisci uniformemente
                for k in range(K):
                    Fs[k, i, j] = F_total_cond_np[i, j] / K

    for k in range(K):
        for i in range(I):
            tot = 0
            for j in range(J):
                tot += F[k,i,j]
            if tot == 0:
                for j in range(J):
                    Fs[k,i,j] = 0
            else: 
                for j in range(J):
                    Fs[k,i,j] = F[k,i,j]/tot
    
    return Fs

