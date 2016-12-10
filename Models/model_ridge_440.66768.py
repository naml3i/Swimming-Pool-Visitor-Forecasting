import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


CROSS_VALIDATION_ITER = 50


# Reading training, test, weather data and submission file
train = pd.read_csv('../Data/train_final.csv')
test = pd.read_csv('../Data/test_final.csv')

# Extract validation data from training data
def create_validation_data(training_data):
    train_validation, test_validation = train_test_split(training_data,
                                                         test_size=0.3)
    return train_validation, test_validation


# Creating ridge regression object (alpha = 0.5, max_iter=1000000,
# tol=0.0000001) and predictors
model = linear_model.Ridge(alpha=0.4, fit_intercept=True, normalize=True,
                           max_iter=1000000, tol=0.0000001, solver='auto')
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
RMSE_by_base = RMSE / np.mean(train['visitors_pool_total'])
print 'Mean RMSE: %.4f' % RMSE_by_base

# Model 10: Predictions = Ridge regression (with tuned params) using sparse
# training set plus derived variables - 'day of month', 'day of the week',
# 'month of year' plus weather data (excluding missing data)
# [SCORE: 440.66768]
model.fit(train[predictors], train['visitors_pool_total'])
test['visitors_pool_total'] = model.predict(test[predictors])
test['visitors_pool_total'] = test['visitors_pool_total'].astype(int)

# Key drivers/drainers
coefficients = dict()
for predictor in predictors:
    coefficients[predictor] = list(model.coef_)[predictors.index(predictor)]
coefficients = dict((k, '%.1f' % v) for k, v in coefficients.iteritems()
                    if abs(v) > 9e-2)
print coefficients
{'snow_height_DWD': '-4.8', 'temperature_UniOS': '10.9', 'month': '-22.3',
 'precipitation_DWD': '3.1', 'price_adult_90min': '211.4', 'event': '64.3',
 'wind_speed_max_UniOS': '1.0', 'global_solar_radiation_UniOS': '0.2',
 'school_holiday': '160.9', 'air_pressure_UniOS': '-1.1', 'price_reduced_90min': '185.6',
 'day': '-0.8', 'bank_holiday': '27.9', 'wind_direction_NW': '-34.3',
 'wind_direction_N': '-61.6', 'wind_direction_SE': '7.9', 'wind_direction_E': '57.0',
 'day_of_week': '72.9', 'air_humidity_UniOS': '-0.7', 'wind_direction_NE': '-119.8',
 'sloop_days_since_opening': '0.3', 'wind_direction_W': '-6.1', 'wind_direction_SW': '-6.0',
 'sportbad_closed': '205.4', 'wind_speed_avg_UniOS': '-0.9'}


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
plt.savefig('../Plots/ridge1_actual_vs_predictions.png')

# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_ridge1_ ' +
                        str(RMSE_by_base) + '.csv', index=False)
