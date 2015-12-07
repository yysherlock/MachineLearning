"""
Using Dato
"""
import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')
train_data, test_data = sales.random_split(.8, seed = 0)

### Section 1:
# Let's compute the mean of the House Prices in King County in 2 different ways.
prices = sales['price'] # extract the price column of the sales SFrame -- this is now an SArray

# recall that the arithmetic average (the mean) is the sum of the prices divided by the total number of houses:
sum_prices = prices.sum()
num_houses = prices.size() # when prices is an SArray .size() returns its length
avg_price_1 = sum_prices/num_houses
avg_price_2 = prices.mean() # if you just want the average, the .mean() function
print "average price via method 1: " + str(avg_price_1)
print "average price via method 2: " + str(avg_price_2)

# if we want to multiply every price by 0.5 it's a simple as:
half_prices = 0.5*prices
# Let's compute the sum of squares of price. We can multiply two SArrays of the same length elementwise also with *
prices_squared = prices*prices
sum_prices_squared = prices_squared.sum() # price_squared is an SArray of the squares and we want to add them up.
print "the sum of price squared is: " + str(sum_prices_squared)


### Section 2:
def simple_linear_regression(input_feature, output):
    # compute the mean of  input_feature and output
    x_sum = input_feature.sum()
    x_mean = input_feature.mean()
    y_mean = output.mean()
    # compute the product of the output and the input_feature and its mean
    xy = input_feature * output
    xy_sum = xy.sum()
    xy_mean = xy.mean()
    # compute the squared value of the input_feature and its mean
    xsquare = input_feature * input_feature
    xsquare_sum = xsquare.sum()
    xsquare_mean = xsquare.mean()
    # use the formula for the slope
    slope = (xy_sum - y_mean*x_sum) / (xsquare_sum - x_mean*x_sum)
    # use the formula for the intercept
    intercept = y_mean - slope*x_mean
    return (intercept, slope)

# test simple_linear_regression function
"""
We can test that our function works by passing it something
where we know the answer.
In particular we can generate a feature and then put the output exactly
on a line: output = 1 + 1*input_feature
then we know both our slope and intercept should be 1.
"""
test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)

sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = input_feature * slope + intercept
    return predicted_values

### predict values
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

### Calculate RSS
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_values = get_regression_predictions(input_feature, intercept, slope)
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = output - predicted_values
    # square the residuals and add them up
    RSS = (residuals * residuals).sum()
    return(RSS)

print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) # should be 0.0
rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)
