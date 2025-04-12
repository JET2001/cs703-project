from typing import List
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import DataProcessing
from src.solver import solve_robust_classification
from src.evaluate import get_auc, ce_loss, plot_auc
from src.utils import Dataset
from tqdm import tqdm
from src.uncertainty import BoxUncertainty, BenignNormUncertainty, MalignantNormUncertainty
import time
import os  
if __name__ == '__main__':
    try:
        os.mkdir("./out")
    except FileExistsError as e:
        pass
    
    
    time_str = f"{time.time_ns() // 1e6}"
    # hyperparams = {
    #     'data_path': "./data_files/breast-cancer-data.csv",
    #     'test_size': 0.3,
    #     'features': ['mean_fractal_dimension', 'label'],
    #     'random_state': 42,
    #     'uncertainty_features_list': []
    # }
    # data_proc = DataProcessing(hyperparams)
    # dataset : Dataset = data_proc.dataset
    # uncertainty_info : List = data_proc.uncertainty_info
    
    # theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    # loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    # print("ce_loss = ", loss)
    # auc, (tpr, fpr) = get_auc(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy() )
    # plot_auc(tpr, fpr, auc, title = "Drop", time_str = time_str)
    
    #########################################################################
    # time_str = f"{time.time_ns() // 1e6}"
    # hyperparams = {
    #     'data_path': "./data_files/breast-cancer-data.csv",
    #     'test_size': 0.3,
    #     'features': ['mean_fractal_dimension', 'mean_texture', 'label'],
    #     'random_state': 42,
    #     'uncertainty_features_list': [
    #         (BoxUncertainty, {
    #             'col_name': 'mean_texture',
    #             'step': 2,
    #             'low' : -3,
    #             'high': 4
    #         })
    #     ]
    # }
    # data_proc = DataProcessing(hyperparams)
    # dataset : Dataset = data_proc.dataset
    # uncertainty_info : List = data_proc.uncertainty_info
    
    # theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    # loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    # print("ce_loss = ", loss)
    # auc, (tpr, fpr) = get_auc(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy() )
    # plot_auc(tpr, fpr, auc, title = "Box Uncertainty on Mean Perimeter", time_str = time_str)
    
    #########################################################################
    hyperparams = {
        'data_path': "./data_files/breast-cancer-data.csv",
        'test_size': 0.3,
        'features': ['mean_area', 'mean_perimeter', 'label'],
        'random_state': 42,
        'uncertainty_features_list': [
            (BoxUncertainty, {
                'col_name': 'mean_area',
                'step': 0.5,
                'low' : -3,
                'high': 3
            }),
            (BoxUncertainty, {
                'col_name': 'mean_perimeter',
                'step': 0.5,
                'low' : -3,
                'high': 3
            }),
            (BenignNormUncertainty, {
                'col_name' : None, # deliberate
                'data_path': "./data_files/breast-cancer-data.csv",
                'test_size': 0.3,
                'random_state': 42,
            }),
            (MalignantNormUncertainty, {
                'col_name' : None, # deliberate
                'data_path': "./data_files/breast-cancer-data.csv",
                'test_size': 0.3,
                'random_state': 42,
            })
        ]
    }
    data_proc = DataProcessing(hyperparams)
    dataset : Dataset = data_proc.dataset
    uncertainty_info : List = data_proc.uncertainty_info
    
    theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    print("ce_loss = ", loss)
    auc, (tpr, fpr) = get_auc(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy() )
    plot_auc(tpr, fpr, auc, title = "Perimeter (Box) intersect Irregularity Norm Ball", time_str = time_str)

