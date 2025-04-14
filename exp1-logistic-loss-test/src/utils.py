from collections import namedtuple

Dataset = namedtuple('Dataset', ('X_train', 'X_test', 'robust_X_test', 'y_train', 'y_test', 'train_rows', 'test_rows'))

UncertaintyInfo = namedtuple("UncertaintyInfo", ('col_name', 'enc', 'name', 'requires'))