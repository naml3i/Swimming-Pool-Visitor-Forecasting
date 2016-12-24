import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


CROSS_VALIDATION_ITER = 50


# Reading training, test, weather data and submission file
train = pd.read_csv('../Data/train_final.csv')
test = pd.read_csv('../Data/test_final.csv')

# Excluding outliers from data
train = train[train['visitors_pool_total'] < 3500]

# Extract validation data from training data
def create_validation_data(training_data):
    train_validation, test_validation = train_test_split(training_data,
                                                         test_size=0.3)
    return train_validation, test_validation


# Creating elastic net object (alpha = 0.5, lambda = 0.8,  max_iter=1000000,
# tol=0.0000001) and predictors
model = linear_model.ElasticNet(
    alpha=0.5, l1_ratio=0.8, fit_intercept=True, normalize=False,
    max_iter=1000000, tol=0.0000001)
predictors = list(train.columns.values)
predictors.remove('date')
predictors.remove('visitors_pool_total')

# Calculating error (RMSE) for the model
rmse_group = list()
model_counter = 0
for iter in xrange(CROSS_VALIDATION_ITER):
    model_counter += 1
    train_validation, test_validation = create_validation_data(train)
    model.fit(train_validation[predictors],
              train_validation['visitors_pool_total'])

    coefficients = dict()
    for predictor in predictors:
        coefficients[predictor] = list(
            model.coef_)[predictors.index(predictor)]

    print
    print 'Model #', str(model_counter)
    print 'Coefficients: \n', coefficients
    rmse = np.sqrt(np.mean((model.predict(test_validation[predictors])
                            - test_validation['visitors_pool_total']) ** 2))
    print 'RMSE: %.2f' % rmse
    print 'R-sqr: %.4f' % model.score(test_validation[predictors],
                                      test_validation['visitors_pool_total'])
    rmse_group.append(rmse)

RMSE = np.mean(rmse_group)
print 'Mean RMSE: %.4f' % RMSE
RMSE_by_base = RMSE / np.mean(train['visitors_pool_total'])
print 'Mean RMSE by base: %.4f' % RMSE_by_base

# Model 11: Predictions = Elastic Net (with tuned params) using sparse
# training set plus derived variables - 'day of month', 'day of the week',
# 'month of year' plus weather data (excluding missing data)
# [SCORE: 474.45379]
model.fit(train[predictors], train['visitors_pool_total'])
test['visitors_pool_total'] = model.predict(test[predictors])
test['visitors_pool_total'] = test['visitors_pool_total'].astype(int)

# Key drivers/drainers
coefficients = dict()
for predictor in predictors:
    coefficients[predictor] = list(model.coef_)[predictors.index(predictor)]
coefficients = dict((k, '%.4f' % v) for k, v in coefficients.iteritems()
                    if abs(v) > 0)
print coefficients
{'snow_height_DWD': '-7.3008', 'temperature_UniOS': '19.5911', 'month': '-38.6450',
 'precipitation_DWD': '1.6960', 'price_adult_90min': '30.2586', 'event': '16.2751',
 'wind_speed_max_UniOS': '1.3755', 'global_solar_radiation_UniOS': '-0.1852',
 'school_holiday': '216.4700', 'air_pressure_UniOS': '-0.3869', 'price_reduced_90min': '61.7162',
 'day': '-1.9998', 'bank_holiday': '16.1091', 'wind_direction_NW': '-12.3333',
 'wind_direction_N': '-6.0809', 'wind_direction_SE': '-4.5980', 'wind_direction_E': '68.3650',
 'day_of_week': '99.2902', 'air_humidity_UniOS': '0.8913', 'wind_direction_NE': '-28.6878',
 'sloop_days_since_opening': '0.8267', 'wind_direction_W': '-1.1343',
 'sportbad_closed': '84.8837', 'wind_speed_avg_UniOS': '-1.5931'}


# Plot cross-validated predictions
fig, ax = plt.subplots()
train_validation, test_validation = create_validation_data(train)
y = train_validation['visitors_pool_total']
train_validation.drop(['date', 'visitors_pool_total'], axis=1, inplace=True)
predicted = cross_val_predict(model, train_validation, y, cv=10)
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.savefig('../Plots/elastic_net1_actual_vs_predictions.png')

# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_elastic_net1_ ' +
                        str(RMSE_by_base) + '.csv', index=False)
