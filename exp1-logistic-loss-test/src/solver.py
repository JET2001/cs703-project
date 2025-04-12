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
        if item.col_name is None and item.name == "Norm": 
            continue
        assert item.col_name in dataset.X_train.columns, item.col_name
            
    theta = cp.Variable(d)
    intercept = cp.Variable(1)
    z = cp.Variable(n) # worst case margin
    epi_log_loss = cp.Variable(n) # epigraph of log loss
    constraints = [z >= 0]
    df = dataset.X_train
    A = dataset.X_train.to_numpy()
    b = dataset.y_train.to_numpy()
    train_row_idxs = dataset.train_rows
    colidx_uncertainty = []
    xcols = list(dataset.X_train.columns)
    for item in uncertainty_info:
        if item.col_name is None: 
            continue
        col_idx = xcols.index(item.col_name)
        colidx_uncertainty.append(col_idx)
    for i in range(n):
        a = A[i]
        if colidx_uncertainty == []:
            constraints += [epi_log_loss[i] >= cp.log_sum_exp(cp.hstack([0, -(b[i] * (theta @ A[i] + intercept))]))]
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
                # print("uncertainty_enc = ", uncertainty_enc)
                # print("x_Enc = ", x_enc)
                if item.name == 'Box':
                    G_constraints_i += [
                        y_loc[col_idx] <= uncertainty_enc[x_enc], 
                        y_loc[col_idx] >= uncertainty_enc[x_enc-1]
                    ]
                else:
                    raise NotImplementedError
                
            for item in uncertainty_info:
                if item.name != 'Norm': 
                    continue
                features = item.requires
                norm_cols = []
                for col_name in features:
                    col_idx = xcols.index(col_name)
                    norm_cols.append(col_idx)
                uncertainty_enc = item.enc
                mean_vec = uncertainty_enc['center']
                radius_train = uncertainty_enc['radius_train']
                # print(df.index.tolist()[i]
                # print("train index at i:", train_row_idxs[i])
                # print("is in radius_train?", train_row_idxs[i] in radius_train)
                if train_row_idxs[i] in radius_train:
                    ri = radius_train[train_row_idxs[i]]
                    G_constraints_i += [
                        cp.norm(y_loc[norm_cols] - mean_vec, 2) <= ri
                    ]
                    
            Gi = SuppFunc(y_loc, G_constraints_i)(-b[i] * theta)
            constraints.append(z[i] <= b[i] * intercept - Gi)
            
            # these constraints are basically epi log_loss of sample i >= log (1 + exp(-z[i])) = log(exp(0) + exp(-z[i]))
            constraints += [epi_log_loss[i] >= cp.log_sum_exp(cp.hstack([0, -z[i]]))]
            
    obj = cp.Minimize(cp.sum(epi_log_loss))
    prob = cp.Problem(obj, constraints)

    prob.solve(solver = cp.SCS, verbose=True)
    
    print("theta = ", theta.value)
    print("intercept =  ", intercept.value[0])
    return theta.value, intercept.value[0]