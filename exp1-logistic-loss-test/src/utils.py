from collections import namedtuple

Dataset = namedtuple('Dataset', ('X_train', 'X_test', 'y_train', 'y_test'))

UncertaintyInfo = namedtuple("UncertaintyInfo", ('col_name', 'enc', 'name'))