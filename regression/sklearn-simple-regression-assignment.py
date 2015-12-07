
import os,sys
from os.path import dirname, join
import csv
import numpy as np
from sklearn.datasets.base import Bunch

module_path = dirname('__file__')
data_path = join(module_path,'data')
train_data_path = join(data_path,'kc_house_train_data.csv')
test_data_path = join(data_path,'kc_house_test_data.csv')

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, \
'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, \
'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, \
'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
dt = np.dtype([ (label, np.dtype(t)) for label,t in dtype_dict.items()])
#print dt
#sys.exit(1)

def load_simple_csv(filename, target_col = -1):
    """ feature is a list of cols
        target is a list of cols
    """
    #target_names = []
    #target = []
    #features = []
    n_samples = -1
    with open(filename) as csv_file:
        for line in csv_file:
            n_samples += 1

    with open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data_names = np.array(next(data_file))
        #print target_names.shape
        feature_names = np.delete(data_names,target_col) # 1 target , other cols are all features
        n_features = feature_names.shape[0]

        target = np.empty((n_samples,), dtype = np.dtype(float))
        features = np.empty((n_samples, n_features))
        type_list = [ (label, np.dtype(t)) for label,t in dtype_dict.items() ]
        type_list.pop(target_col)
        dt = np.dtype(type_list)
        # print len(dt)
        for i, item in enumerate(data_file):
            # print item,len(item)
            t = item.pop(target_col)
            target[i] = np.asarray(t, dtype = np.float64)
            features[i] = np.asarray(item, dtype = dt)

    return Bunch(data=features, target=target,
                 target_names=None, # precit problem
                 DESCR=None,
                 feature_names=feature_names)

train_data = load_simple_csv(train_data_path, 2)
print train_data.feature_names
print train_data.data
