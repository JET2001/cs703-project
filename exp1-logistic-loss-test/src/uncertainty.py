from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import Dataset

import numpy as np
import bisect
import copy
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
    def requires_features(self)->List[str]:
        return []

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

        res_test = []
        X_test_subset = dataset.X_test[[col_name]].copy()
        
        for x in X_test_subset.values:
            bin_no = bisect.bisect_left(self.bins, x)
            res_test.append(bin_no)
        X_test_transformed = dataset.robust_X_test.copy()
        X_test_transformed[col_name] = res_test
        
        self.transformed = Dataset(X_train_transformed, dataset.X_test, X_test_transformed, dataset.y_train, dataset.y_test, dataset.train_rows, dataset.test_rows)

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return self.bins
    
    def get_name(self):
        return "Box"

class IrregularityNormUncertainty(BaseUncertainty):
    '''
    This uncertainty set adds rows to the dataset.
    Requires mean_perimeter, and mean_compactness to be in the dataset.
    We form a vector x = (mean_perimeter, mean_compactness), and for each point
    we define a L2 norm ball around the baricenter x\bar = (\bar mean_perimeter, \bar mean_compactness), where x\bar is the baricenter of mean_perimeter and mean_compactness, as a measure of deviation from the mean.
    
    We remove the mean compactness, and mean perimeter features after adding this uncertainty set. 
    
    Also, the test set values are being scaled using the baricenter of the data in the train set. 
    
    Fundamentally, these two variables are related via:
    Compactness ~ Perimeter^2 / Area. 
    '''
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        self.col_name = params.get('col_name', None)
        assert self.col_name is None, "col_name should be left to None for Norm Uncertainty Sets"
        self.data_path = params.get('data_path')
        assert self.data_path is not None
        self.test_size = params.get('test_size')
        self.features = params.get('features')
        self.random_state = params.get('random_state')
        assert self.test_size is not None
        df = pd.read_csv(self.data_path)[['mean_radius', 'mean_texture', 'label']]
        df['label'] = [2*y -1 for y in df['label']]
        X = df.copy().drop('label', axis = 1)
        # X = df.copy()
        y = df.label
        
        X_train, X_test, _, _ = train_test_split(
            X, y, 
            test_size = self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        sc = StandardScaler().fit(X_train) # fit only on the benign class in the train set.
        X_train_norm = pd.DataFrame(sc.transform(X_train), columns = X.columns)
        X_test_norm = pd.DataFrame(sc.transform(X_test), columns = X.columns)
        
        Xbar_train = X_train_norm.mean()
        self.res_train = np.linalg.norm(X_train_norm.values - Xbar_train.values, axis = 1)
        self.res_test = np.linalg.norm(X_test_norm.values - Xbar_train.values, axis = 1)
        self.transformed = copy.deepcopy(dataset)
        # self.transformed.X_train[self.col_name] = res_train
        # self.transformed.X_test[self.col_name] = res_test
        # # self.transformed.X_train['Irregularity'] = res_train
        # # self.transformed.X_test['Irregularity'] = res_test
        
        self.center = Xbar_train.values

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return {'center': self.center, 'radius_train' : self.res_train, 'radius_test': self.res_test }
    
    def get_name(self):
        return "Norm"
    
    def requires_features(self):
        return ["mean_radius", "mean_texture"]

class BenignNormUncertainty(BaseUncertainty):
    '''
    Norm to the benign reference point.
    '''
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        self.col_name = params.get('col_name', None)
        assert self.col_name is None, "col_name should be left to None for Norm Uncertainty Sets"
        self.data_path = params.get('data_path')
        assert self.data_path is not None
        self.test_size = params.get('test_size')
        self.features = params.get('features')
        self.random_state = params.get('random_state')
        assert self.test_size is not None
        df = pd.read_csv(self.data_path)[['mean_perimeter', 'mean_area', 'label']]
        df['label'] = [2*y -1 for y in df['label']]
        X = df.copy().drop('label', axis = 1)
        # X = df.copy()
        y = df.label
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size = self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        sc = StandardScaler().fit(X_train) # fit only on the benign class in the train set.
        X_train_norm = pd.DataFrame(sc.transform(X_train), columns = X.columns)
        X_test_norm = pd.DataFrame(sc.transform(X_test), columns = X.columns)
        
        df_train = X_train_norm.merge(y_train, how = 'inner', left_index = True, right_index = True)
        df_train = df_train[df_train.label == -1].drop('label', axis = 1)
        # self.res_train = {}
        self.center = df_train.mean().values
        # print("self.center = ", self.center)
        
        res_train = np.linalg.norm(df_train.values - self.center, axis = 1)
        # self.res_test = np.linalg.norm(X_test_norm.values - self.center, axis = 1)
        self.res_train = {idx: x for idx, x in zip (df_train.index.tolist(), res_train)}
        print(self.res_train)
        
        
        self.transformed = copy.deepcopy(dataset)
        # self.transformed.X_train[self.col_name] = res_train
        # self.transformed.X_test[self.col_name] = res_test
        # # self.transformed.X_train['Irregularity'] = res_train
        # # self.transformed.X_test['Irregularity'] = res_test

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return {'center': self.center, 'radius_train' : self.res_train }
    
    def get_name(self):
        return "Norm"
    
    def requires_features(self):
        return ["mean_area", "mean_perimeter"]

class MalignantNormUncertainty(BaseUncertainty):
    '''
    Norm to the malignant reference point.
    '''
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        self.col_name = params.get('col_name', None)
        assert self.col_name is None, "col_name should be left to None for Norm Uncertainty Sets"
        self.data_path = params.get('data_path')
        assert self.data_path is not None
        self.test_size = params.get('test_size')
        self.features = params.get('features')
        self.random_state = params.get('random_state')
        assert self.test_size is not None
        df = pd.read_csv(self.data_path)[['mean_perimeter', 'mean_area', 'label']]
        df['label'] = [2*y -1 for y in df['label']]
        X = df.copy().drop('label', axis = 1)
        # X = df.copy()
        y = df.label
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size = self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        sc = StandardScaler().fit(X_train) # fit only on the benign class in the train set.
        X_train_norm = pd.DataFrame(sc.transform(X_train), columns = X.columns)
        X_test_norm = pd.DataFrame(sc.transform(X_test), columns = X.columns)
        
        df_train = X_train_norm.merge(y_train, how = 'inner', left_index = True, right_index = True)
        df_train = df_train[df_train.label == 1].drop('label', axis = 1)
        # self.res_train = {}
        self.center = df_train.mean().values
        
        res_train = np.linalg.norm(df_train.values - self.center, axis = 1)
        res_test = np.linalg.norm(X_test_norm.values - self.center, axis = 1)
        self.res_train = {idx: x for idx, x in zip (df_train.index.tolist(), res_train)}
        # self.res_test = {idx : x for idx, x in zip (df)}
        
        self.transformed = copy.deepcopy(dataset)
        # self.transformed.X_train[self.col_name] = res_train
        # self.transformed.X_test[self.col_name] = res_test
        # # self.transformed.X_train['Irregularity'] = res_train
        # # self.transformed.X_test['Irregularity'] = res_test

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return {'center': self.center, 'radius_train' : self.res_train }
    
    def get_name(self):
        return "Norm"
    
    def requires_features(self):
        return ["mean_area", "mean_perimeter"]

class PerimeterUncertainty(BaseUncertainty):
    
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        self.col_name = 'mean_perimeter'
        self.data_path = params.get('data_path')
        assert self.data_path is not None
        df = pd.read_csv(self.data_path, index_col = False)[['mean_perimeter', 'mean_radius']]
        df['perimeter_ideal'] = [2 * np.pi * r for r in df.mean_radius]
        # area_ideal = df.mean_radius
        self.enc = [abs(x - y) for x, y in zip(df.mean_perimeter, df.perimeter_ideal)]
        self.perimeter_ideal = df.perimeter_ideal
        
        
        self.transformed = copy.deepcopy(dataset)
        # self.transformed.X_train[self.col_name] = res_train
        # self.transformed.X_test[self.col_name] = res_test
        # # self.transformed.X_train['Irregularity'] = res_train
        # # self.transformed.X_test['Irregularity'] = res_test

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return {'enc' : self.enc, 'ideal': self.perimeter_ideal } 
    
    def get_name(self):
        return "L1-Norm"
    
    def requires_features(self):
        return ["mean_perimeter"]
    
class AreaUncertainty(BaseUncertainty):
    
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        self.col_name = 'mean_radius'
        self.data_path = params.get('data_path')
        assert self.data_path is not None
        df = pd.read_csv(self.data_path)[['mean_area', 'mean_radius']]
        df['area_ideal'] = [np.pi * r * r for r in df.mean_radius]
        self.area_ideal = df.area_ideal
        self.enc = [abs(x - y) for x, y in zip(df.mean_area, df.area_ideal)]
        
        self.transformed = copy.deepcopy(dataset)

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return {'enc' : self.enc, 'ideal': self.area_ideal } 
    
    def get_name(self):
        return "L1-Norm"
    
    def requires_features(self):
        return ["mean_area"]

class CircleUncertainty(BaseUncertainty):
    
    def __init__(self, dataset: Dataset, params: Dict):
        super().__init__(params)
        self.col_name = None
        self.data_path = params.get('data_path')
        assert self.data_path is not None
        df = pd.read_csv(self.data_path)[['mean_area','mean_perimeter', 'mean_radius']]
        df['area_ideal'] = [np.pi * r * r for r in df.mean_radius]
        df['perimeter_ideal'] = [2 * np.pi * r for r in df.mean_radius]
        self.area_ideal = df.area_ideal
        self.perimeter_ideal = df.perimeter_ideal
        
        data = df[['mean_area','mean_perimeter']].values
        self.ideal_data = df[['area_ideal', 'perimeter_ideal']].values
        self.enc = np.linalg.norm(data - self.ideal_data, ord = 1, axis = 1)
        
        # self.ideal_data = ideal_data
        self.transformed = copy.deepcopy(dataset)

    def get_transformed_dataset(self):
        return self.transformed
    
    def get_uncertainty_enc(self):
        return {'enc' : self.enc, 'ideal': self.ideal_data } 
    
    def get_name(self):
        return "Norm"
    
    def requires_features(self):
        return ["mean_area", "mean_perimeter"]