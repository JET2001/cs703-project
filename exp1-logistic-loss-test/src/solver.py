import numpy as np 
import cvxpy as cp
import pandas as pd
import dsp                                     # disciplined saddle point programming
from typing import Dict, List, Tuple, Union
from .utils import Dataset, UncertaintyInfo
from .uncertainty import BaseUncertainty
from cvxpy.transforms.suppfunc import SuppFunc # Import SuppFunc to transform support function constraints

def solve_robust_classification(
        dataset: Dataset,
        uncertainty_info: List[UncertaintyInfo],
):
    '''
    uncertainty_info is encoded as (column index, encoding map)

    A - design matrix
    b - labels
    '''
    n, d = dataset.X_train.to_numpy().shape
    assert dataset.y_train.to_numpy().shape == (n,)
    for item in uncertainty_info:
        assert item[0] in dataset.X_train.columns, item[0]
            
    theta = cp.Variable(d)
    intercept = cp.Variable(1)
    z = cp.Variable(n) # worst case margin
    epi_log_loss = cp.Variable(n) # epigraph of log loss
    constraints = [z >= 0]
    A = dataset.X_train.to_numpy()
    b = dataset.y_train.to_numpy()
    colidx_uncertainty = []
    xcols = list(dataset.X_train.columns)
    for item in uncertainty_info:
        col_idx = xcols.index(item.col_name)
        colidx_uncertainty.append(col_idx)
    for i in range(n):
        a = A[i]
        if colidx_uncertainty == []:
            # z[i] >= b[i] * (theta @ A[i] + intercept)
            constraints += [epi_log_loss[i] >= cp.log_sum_exp(cp.hstack([0, -(b[i] * (theta @ A[i] + intercept))]))]
            # constraints.append(cp.abs(cp.sum(cp.multiply(a,x)) + intercept - b[i]) <= z[i])
        else:
            y_loc = dsp.LocalVariable(d)
            G_constraints_i = []

            for col_idx in range(d):
                if col_idx not in colidx_uncertainty:
                    G_constraints_i.append(y_loc[col_idx] == a[col_idx])
            # Consider the uncertainty sets
            for col_idx, item in zip(colidx_uncertainty, uncertainty_info):
                uncertainty_enc = item.enc
                x_enc = int(A[i, col_idx].item())
                print("x_Enc = ", x_enc)
                if item.name == 'Box':
                    G_constraints_i += [
                        y_loc[col_idx] <= uncertainty_enc[x_enc], y_loc[col_idx] >= uncertainty_enc[x_enc-1]
                    ]
                else:
                    raise NotImplementedError
                
            Gi = SuppFunc(y_loc, G_constraints_i)(-b[i] * theta)
            constraints.append(z[i] <= b[i] * intercept - Gi)
            
            # these constraints are basically epi log_loss of sample i >= log (1 + exp(-z[i])) = log(exp(0) + exp(-z[i]))
            constraints += [epi_log_loss[i] >= cp.log_sum_exp(cp.hstack([0, -z[i]]))]
            
    obj = cp.Minimize(cp.sum(epi_log_loss))
    prob = cp.Problem(obj, constraints)

    prob.solve(solver = cp.ECOS, verbose=True)
    return theta.value, intercept.value[0]