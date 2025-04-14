import numpy as np 
import cvxpy as cp
import pandas as pd
import dsp                                     # disciplined saddle point programming
from typing import Dict, List, Tuple, Union
from .utils import Dataset, UncertaintyInfo
from .uncertainty import BaseUncertainty
from cvxpy.transforms.suppfunc import SuppFunc # Import SuppFunc to transform support function constraints
from tqdm import tqdm
import statistics


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
    constraints = []
    A = dataset.X_train.to_numpy()
    b = dataset.y_train.to_numpy()
    train_row_idxs = dataset.train_rows
    colidx_uncertainty = []
    xcols = list(dataset.X_train.columns)
    for item in uncertainty_info:
        print("item.name = ", item.name)
        print("item.enc = ", item.enc)
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
                # print("uncertainty_enc = ", uncertainty_enc)
                # print("x_Enc = ", x_enc)
                if item.name == 'Box':
                    uncertainty_enc = item.enc
                    x_enc = int(A[i, col_idx].item())
                    try:
                        G_constraints_i += [
                            y_loc[col_idx] <= uncertainty_enc[x_enc], 
                            y_loc[col_idx] >= uncertainty_enc[x_enc-1]
                        ]
                    except IndexError as e:
                        print("col_idx = ", col_idx, "xenc= ", x_enc)
                        print(e)
                        
                elif item.name == 'L1-Norm':
                    uncertainty_enc = item.enc['enc']
                    ideal_enc = item.enc['ideal']
                    sample_i = train_row_idxs[i]
                    xval = uncertainty_enc[sample_i]
                    xval_ideal = ideal_enc[sample_i]
                    G_constraints_i += [
                        cp.abs(y_loc[col_idx] -  xval_ideal) <= xval
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
                rhs = uncertainty_enc['enc']
                ideal_data = uncertainty_enc['ideal']
                # print(df.index.tolist()[i]
                # print("train index at i:", train_row_idxs[i])
                # print("is in radius_train?", train_row_idxs[i] in radius_train)
                ai = train_row_idxs[i]
                G_constraints_i += [
                    cp.norm(y_loc[norm_cols] - ideal_data[ai], 1) <= rhs[ai]
                ]
                # if train_row_idxs[i] in radius_train:
                #     ri = radius_train[train_row_idxs[i]]
                #     G_constraints_i += [
                #         cp.norm(y_loc[norm_cols] - mean_vec, 2) <= ri
                #     ]
                    
            Gi = SuppFunc(y_loc, G_constraints_i)(-b[i] * theta)
            constraints.append(z[i] <= b[i] * intercept - Gi)
            
            # these constraints are basically epi log_loss of sample i >= log (1 + exp(-z[i])) = log(exp(0) + exp(-z[i]))
            constraints += [epi_log_loss[i] >= cp.log_sum_exp(cp.hstack([0, -z[i]]))]
            
    obj = cp.Minimize(cp.sum(epi_log_loss))
    prob = cp.Problem(obj, constraints)

    prob.solve(solver = cp.SCS, max_iters = 10000, verbose=True)
    
    print("train ce_loss = ", prob.value/n)
    print("theta = ", theta.value)
    print("intercept =  ", intercept.value[0])
    return theta.value, intercept.value[0]

def evaluate_ce_loss(
    dataset: Dataset,
    uncertainty_info: List[UncertaintyInfo],
    theta: np.ndarray,
    intercept: np.ndarray,
    n_samples: int = 100,
):
    
    A = dataset.robust_X_test.to_numpy()
    b = dataset.y_test.to_numpy()
    xcols = dataset.robust_X_test.columns.tolist()
    test_rows = dataset.test_rows
    
    n, d = A.shape
    rng = np.random.Generator(np.random.PCG64(seed = 42))
    losses = []
    for i in tqdm(range(n), desc = "Performing evaluation"):
        for _ in range(n_samples):
            loss_i = 0.0
            ai = A[i,:].copy()
            # print(ai, ai.shape)
            for item in uncertainty_info:
                if item.name == 'Box':
                    col_idx = xcols.index(item.col_name)
                    enc = item.enc
                    # print(enc[int(ai[col_idx])-1],  enc[int(ai[col_idx])])
                    ai[col_idx] = rng.uniform(low = enc[int(ai[col_idx])-1], high = enc[int(ai[col_idx])])
            loss_i = max(loss_i, np.log(1 + np.exp(- (b[i] * (theta @ ai + intercept)))))
            # print("loss_i = ", loss_i)
        losses.append(loss_i)
    return statistics.mean(losses)       