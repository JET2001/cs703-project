from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import pandas as pd
from .utils import Dataset

import numpy as np
import bisect

class BaseUncertainty(ABC):

    def __init__(self, params: Dict):
        pass

    @abstractmethod
    def get_transformed_dataset(self)->Dataset:
        '''
        Returns a transformed dataset with uncertainty already added to the samples
        '''
        pass
    
    @abstractmethod
    def get_uncertainty_enc(self)->Union[Dict, List]:
        '''
        return a Dictionary type object / list type object that accepts the encoding by the uncertainty set, and output 
        the actual value given by this uncertainty set
        '''
        pass
    @abstractmethod
    def get_name(self)->str:
        '''
        return a name that is used by the solver to generate constraints
        '''
        pass

class BoxUncertainty(BaseUncertainty):
    '''
    Given an encoding from this uncertainty set, the bounding coordinates is given by [bin_no-1,bin_no]
    '''
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        col_name = params.get('col_name')
        assert col_name in dataset.X_train.columns
        X_train_subset = dataset.X_train[[col_name]].copy()
        print(X_train_subset)

        self.step = params.get('step')
        self.low = params.get('low')
        self.high = params.get('high')
        self.bins = np.arange(self.low, self.high, self.step).tolist()
        res = []
        for x in X_train_subset.values:
            bin_no = bisect.bisect_left(self.bins, x)
            res.append(bin_no)
        X_train_transformed = dataset.X_train.copy()
        X_train_transformed[col_name] = res
        self.transformed = Dataset(X_train_transformed, dataset.X_test, dataset.y_train, dataset.y_test)

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return self.bins
    
    def get_name(self):
        return "Box"