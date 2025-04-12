import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from math import ceil, floor
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import Dataset, UncertaintyInfo
from .uncertainty import BaseUncertainty

class DataProcessing:
    '''
    This class reads the dataset and generates uncertainty lists
    '''
    def __init__(self, hyperparams: Dict):

        self.data_path = hyperparams.get('data_path')
        assert self.data_path is not None
        self.test_size = hyperparams.get('test_size')
        self.features = hyperparams.get('features')
        self.random_state = hyperparams.get('random_state')
        assert self.test_size is not None
        assert isinstance(self.features, list), type(self.features)
        df = pd.read_csv(self.data_path)[self.features]
        # scale to +-1 encoding
        df['label'] = [2*y -1 for y in df['label']]
        print(df.head(5))
        X = df.drop('label', axis = 1)
        y = df.label

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = self.test_size, 
                                                        random_state=self.random_state,
                                                        stratify=y
                                                        )
        train_rows = X_train.index.tolist()
        test_rows = X_test.index.tolist()
        
        sc = StandardScaler().fit(X_train)
        X_train_norm = pd.DataFrame(np.clip(sc.transform(X_train), a_min = -2, a_max = 2), columns = X_train.columns)
        X_test_norm = pd.DataFrame(np.clip(sc.transform(X_test), a_min = -2, a_max = 2), columns = X_train.columns)
        self._dataset = Dataset(X_train_norm, X_test_norm, y_train, y_test, train_rows, test_rows)
        

        self.uncertainty_features_list = hyperparams.get('uncertainty_features_list', [])
        assert isinstance(self.uncertainty_features_list, list), type(self.uncertainty_features_list)
        self._uncertainty_info: List[UncertaintyInfo] = []
        self.mask_train_df()
        self._dataset.X_train.to_excel("out/analyse-X_train.xlsx")
        pass

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def uncertainty_info(self)->List[UncertaintyInfo]:
        return self._uncertainty_info
    
    def mask_train_df(self):
        for item in self.uncertainty_features_list:
            uncertainty_class, params = item
            uncertainty : BaseUncertainty = uncertainty_class(self._dataset, params)
            self._dataset = uncertainty.get_transformed_dataset()
            self._uncertainty_info.append(UncertaintyInfo(col_name=params.get('col_name'), enc = uncertainty.get_uncertainty_enc(), name = uncertainty.get_name(), requires = uncertainty.requires_features()))
            
# class IrregularityFeatureEng(DataProcessing):
#     '''
#     Requires mean_perimeter, and mean_compactness to be in the dataset.
#     We form a vector x = (mean_perimeter, mean_compactness), and for each point
#     we define a L2 norm ball around the baricenter x\bar = (\bar mean_perimeter, \bar mean_compactness), where x\bar is the baricenter of mean_perimeter and mean_compactness, as a measure of deviation from the mean.
    
#     We remove the mean compactness, and mean perimeter features after adding this uncertainty set. 
    
#     Also, the test set values are being scaled using the baricenter of the data in the train set. 
    
#     Fundamentally, these two variables are related via:
#     Compactness ~ Perimeter^2 / Area. 
#     '''
#     def __init__(self, hyperparams: Dict):
#         super().__init__(hyperparams)
#         self.data_path = hyperparams.get('data_path')
#         assert self.data_path is not None
#         self.test_size = hyperparams.get('test_size')
#         self.features = hyperparams.get('features')
#         self.random_state = hyperparams.get('random_state')
#         assert self.test_size is not None
#         df = pd.read_csv(self.data_path)[['mean_texture', 'mean_perimeter', 'label']]
#         df['label'] = [2*y -1 for y in df['label']]
#         X = df.copy().drop('label', axis = 1)
#         # X = df.copy()
#         y = df.label
    

#         X_train, X_test, _, _ = train_test_split(X, y, 
#                                                         test_size = self.test_size, 
#                                                         random_state=self.random_state,
#                                                         stratify=y
#                                                         )
#         sc = StandardScaler().fit(X_train) # fit only on the benign class in the train set.
#         X_train_norm = pd.DataFrame(sc.transform(X_train), columns = X.columns)
#         X_test_norm = pd.DataFrame(sc.transform(X_test), columns = X.columns)
#         # X_train_norm = X_train_norm.drop('label', axis = 1)
#         # X_test_norm = X_test_norm.drop('label', axis = 1)
        
#         Xbar_train = X_train_norm.mean()
#         res_train = np.linalg.norm(X_train_norm.values - Xbar_train.values, axis = 1)
#         res_test = np.linalg.norm(X_test_norm.values - Xbar_train.values, axis = 1)
#         self._dataset.X_train['Irregularity'] = res_train
#         self._dataset.X_test['Irregularity'] = res_test