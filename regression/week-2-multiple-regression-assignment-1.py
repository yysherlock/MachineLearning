#-*-coding:utf-8-*-

import graphlab

sales = graphlab.SFrame('data/kc_house_data.gl/')
train_data, test_data = sales.random_split(.8, seed = 0)

print train_data.column_names()
example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data, target='price', features=example_features, validation_set=None)
example_weight_summary = example_model.get("coefficients")
example_predictions = example_model.predict(train_data)

# compute RSS
def get_residual_sum_of_squares(model, data, outcome):
    """
    Params
        data: SFrame, including features and target fitted to the given model.
        outcome: SArray
    """
    # First get the predictions
    predictions = model.predict(data)
    # Then compute the residuals/errors
    residuals = outcome - predictions
    # Then square and add them up
    RSS = (residuals*residuals).sum()
    return(RSS)

rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print rss_example_train

# Create some new features: consider transformations of existing features
"""
Next create the following 4 new features as column in both TEST and TRAIN data:
bedrooms_squared = bedrooms*bedrooms
bed_bath_rooms = bedrooms*bathrooms
log_sqft_living = log(sqft_living)
lat_plus_long = lat + long
"""
from math import log
"""
`.apply` can be used in both SFrame and SArray.
It takes a function as input parameter.
"""
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x : x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)
# create the remaining 3 features in both TEST and TRAIN data
type(train_data) # graphlab.data_structures.sframe.SFrame
train_data['bed_bath_rooms']=train_data.apply(lambda x : x['bedrooms']*x['bathrooms'])
test_data['bed_bath_rooms']=test_data.apply(lambda x : x['bedrooms']*x['bathrooms'])

train_data['log_sqft_living']=train_data['sqft_living'].apply(log)
test_data['log_sqft_living']=test_data['sqft_living'].apply(log)

train_data['lat_plus_long']=train_data.apply(lambda x : x['lat'] + x['long'])
test_data['lat_plus_long']=test_data.apply(lambda x : x['lat'] + x['long'])

"""
* Squaring bedrooms will increase the separation
between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4)
since 1^2 = 1 but 4^2 = 16.
Consequently this feature will mostly affect houses with many bedrooms.
* bedrooms times bathrooms gives what's called an "interaction" feature.
It is large when both of them are large.
* Taking the log of squarefeet has the effect of
bringing large values closer together and spreading out small values.
* Adding latitude to longitude is totally non-sensical but we will do it anyway (you'll see why)
"""

# Learn Multiple Models
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']
# Learn the three models: (don't forget to set validation_set = None)
model_1=graphlab.linear_regression.create(train_data, target='price', features=model_1_features, validation_set=None)
model_2=graphlab.linear_regression.create(train_data, target='price', features=model_2_features, validation_set=None)
model_3=graphlab.linear_regression.create(train_data, target='price', features=model_3_features, validation_set=None)
"""
here, graphlab.linear_regression.create maybe use the gradient descent algorightm
to find the optimal parameters (regression coefficients),
otherwise the H^TH in close form solution is not invertible
since`lat+long` feature is linear combination of `lat` and `long`. """

"""
Think about what this means.
增加了一个 bed_bath_rooms 涉及到的 features有 bedrooms 和 bathrooms
当 bedrooms 或 bathroom 增加一点点 相应的 bed_bath_rooms的值会增加很多(因为乘了另一个feature)
而 bed_bath_room的weight是正的
这样当 bathroom增加时 weighted bathroom 和 weighted bed_bath_rooms 都会增加
因而 bathrooms的weight应该会变小 以至于变成了负的
同理， bedrooms的weight也应该变小
没涉及到的feature的weight变化非常小
"""

# Compute the RSS on TRAINING data for each of the three models and record the values:
RSS_train_1 = get_residual_sum_of_squares(model_1, train_data, train_data['price'])
RSS_train_2 = get_residual_sum_of_squares(model_2, train_data, train_data['price'])
RSS_train_3 = get_residual_sum_of_squares(model_3, train_data, train_data['price'])
print RSS_train_1, RSS_train_2, RSS_train_3

# Compute the RSS on TESTING data for each of the three models and record the values:
RSS_test_1 = get_residual_sum_of_squares(model_1, test_data, test_data['price'])
RSS_test_2 = get_residual_sum_of_squares(model_2, test_data, test_data['price'])
RSS_test_3 = get_residual_sum_of_squares(model_3, test_data, test_data['price'])
print RSS_test_1, RSS_test_2, RSS_test_3
