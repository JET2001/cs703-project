import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from math import ceil, floor
from collections import namedtuple
from sklearn.model_selection import train_test_split
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
        self._dataset = Dataset(X_train, X_test, y_train, y_test)

        self.uncertainty_features_list = hyperparams.get('uncertainty_features_list', [])
        assert isinstance(self.uncertainty_features_list, list), type(self.uncertainty_features_list)
        self._uncertainty_info: List[UncertaintyInfo] = []
        self.mask_X_df()
        pass

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def uncertainty_info(self)->List[UncertaintyInfo]:
        return self._uncertainty_info
    
    def mask_X_df(self)->Dataset:
        Xcols = self._dataset.X_train.columns
        for item in self.uncertainty_features_list:
            uncertainty_class, params = item
            uncertainty : BaseUncertainty = uncertainty_class(self._dataset, params)
            self._dataset = uncertainty.get_transformed_dataset()
            self._uncertainty_info.append(UncertaintyInfo(col_name=params.get('col_name'), enc = uncertainty.get_uncertainty_enc(), name = uncertainty.get_name()))