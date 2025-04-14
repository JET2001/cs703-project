from typing import List
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import DataProcessing
from src.solver import solve_robust_classification, evaluate_ce_loss
from src.evaluate import get_auc, ce_loss, plot_auc
from src.utils import Dataset
from tqdm import tqdm
from src.uncertainty import BoxUncertainty, BenignNormUncertainty, MalignantNormUncertainty, AreaUncertainty, PerimeterUncertainty, CircleUncertainty
import time
import os  
if __name__ == '__main__':
    try:
        os.mkdir("./out")
    except FileExistsError as e:
        pass
    results = {}
    ########################################################################
    # No Uncertainty Test Error
    time_str = f"{time.time_ns() // 1e6}"
    hyperparams = {
        'data_path': "./data_files/breast-cancer-data.csv",
        'test_size': 0.3,
        'features': ['mean_fractal_dimension', 'mean_area', 'mean_perimeter', 'label'],
        'random_state': 42,
        'uncertainty_features_list': []
    }
    
    data_proc = DataProcessing(hyperparams)
    dataset : Dataset = data_proc.dataset
    uncertainty_info : List = data_proc.uncertainty_info
    theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    print("ce_loss = ", loss)
    results['No Uncertainty'] = loss
    
    #######################################################################
    # Drop (no Mean Area and Mean Parameters)
    hyperparams = {
        'data_path': "./data_files/breast-cancer-data.csv",
        'test_size': 0.3,
        'features': ['mean_fractal_dimension', 'label'],
        'random_state': 42,
        'uncertainty_features_list': []
    }
    data_proc = DataProcessing(hyperparams)
    dataset : Dataset = data_proc.dataset
    uncertainty_info : List = data_proc.uncertainty_info
    
    theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    print("ce_loss = ", loss)
    # auc, (tpr, fpr) = get_auc(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy() )
    # # plot_auc(tpr, fpr, auc, title = "Drop", time_str = time_str)
    results['Drop'] = loss
    #########################################################################
    # Only Box Uncertainty on the perimeters
    time_str = f"{time.time_ns() // 1e6}"
    hyperparams = {
        'data_path': "./data_files/breast-cancer-data.csv",
        'test_size': 0.3,
        'features': ['mean_fractal_dimension', 'mean_area', 'mean_perimeter', 'label'],
        'random_state': 42,
        'uncertainty_features_list': [
            (BoxUncertainty, {
                'col_name': 'mean_area',
                'step': 500,
                'low' : -500,
                'high': 2500 + 500
            }),
            (BoxUncertainty, {
                'col_name': 'mean_perimeter',
                'step': 20,
                'low' : 40 - 20,
                'high': 200+20
            }),
        ]
    }
    data_proc = DataProcessing(hyperparams)
    dataset : Dataset = data_proc.dataset
    uncertainty_info : List = data_proc.uncertainty_info
    
    theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    print("ce_loss = ", loss)
    # auc, (tpr, fpr) = get_auc(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy() )
    # plot_auc(tpr, fpr, auc, title = "Box Uncertainty on Mean Perimeter", time_str = time_str)
    results['Box'] = loss
    
    #########################################################################
    # Box + L1 Norm uncertainty
    hyperparams = {
        'data_path': "./data_files/breast-cancer-data.csv",
        'test_size': 0.3,
        'features': ['mean_fractal_dimension', 'mean_area', 'mean_perimeter', 'label'],
        'random_state': 42,
        'uncertainty_features_list': [
            (BoxUncertainty, {
                'col_name': 'mean_area',
                'step': 500,
                'low' : -500,
                'high': 2500 + 500
            }),
            (BoxUncertainty, {
                'col_name': 'mean_perimeter',
                'step': 20,
                'low' : 40 - 20,
                'high': 200+20
            }),
            (CircleUncertainty, {
                'data_path':"./data_files/breast-cancer-data.csv"
            })
        ]
    }
    data_proc = DataProcessing(hyperparams)
    dataset : Dataset = data_proc.dataset
    uncertainty_info : List = data_proc.uncertainty_info
    
    theta, intercept = solve_robust_classification(dataset, uncertainty_info)    
    loss = ce_loss(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy())
    print("ce_loss = ", loss)
    results['Box n Norm'] = loss
    # robust_loss = evaluate_ce_loss(dataset, uncertainty_info, theta, intercept)
    # print("robust_loss = ", robust_loss)
    
    # auc, (tpr, fpr) = get_auc(theta, intercept, dataset.X_test.to_numpy(), dataset.y_test.to_numpy() )
    # plot_auc(tpr, fpr, auc, title = "Perimeter (Box) intersect Irregularity Norm Ball", time_str = time_str)
    ##########################################################################
    
    # results_list = []
    percent_list = []
    keys_list = []
    for key in ['Drop', 'Box', 'Box n Norm']:
        percent_list.append((results[key] - results['No Uncertainty'])/ results[key] * 100)
    results_list = [results[x] for x in ['No Uncertainty', 'Drop', 'Box', 'Box n Norm']]
    
     
    plt.figure(figsize = (7,5))
    plt.bar(x = ['No Uncertainty', 'Drop', 'Box', 'Box n Norm'], height = results_list)
    plt.title("Cross Entropy loss on Test Set")
    plt.ylabel("CE Loss")
    plt.savefig(f"out/{time_str}-ce-loss.pdf")
    
    plt.figure(figsize = (7,5))
    plt.bar(x = ['Drop', 'Box', 'Box n Norm'], height = percent_list)
    plt.title(r"% increase in cross entropy loss on Test Set")
    plt.ylabel("% increase in CE loss")
    plt.savefig(f"out/{time_str}-inc-ce-loss.pdf")
