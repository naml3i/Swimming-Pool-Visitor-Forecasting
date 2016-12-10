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


# Creating lasso regression object (alpha = 0.7, max_iter=1000000,
# tol=0.0000001) and predictors
model = linear_model.Lasso(alpha=0.7, fit_intercept=True, normalize=True,
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
RMSE_by_base = RMSE / np.mean(train['visitors_pool_total'])
print 'Mean RMSE: %.4f' % RMSE_by_base

# Model 10: Predictions = Lasso regression (with tuned params) using sparse
# training set plus derived variables - 'day of month', 'day of the week',
# 'month of year' plus weather data (excluding missing data)
# [SCORE: 405.55446]
model.fit(train[predictors], train['visitors_pool_total'])
test['visitors_pool_total'] = model.predict(test[predictors])
test['visitors_pool_total'] = test['visitors_pool_total'].astype(int)

# Key drivers/drainers
coefficients = dict()
for predictor in predictors:
    coefficients[predictor] = list(model.coef_)[predictors.index(predictor)]
coefficients = dict((k, '%.1f' % v) for k, v in coefficients.iteritems()
                    if abs(v) > 9e-2)
# print coefficients
{'price_reduced_90min': '313.7', 'school_holiday': '212.1', 'wind_direction_E': '28.3',
 'temperature_UniOS': '14.6', 'month': '-25.3', 'wind_direction_NE': '-1.9',
 'sloop_days_since_opening': '0.2', 'day_of_week': '91.8', 'sportbad_closed': '131.2'}


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
plt.savefig('../Plots/lasso3_actual_vs_predictions.png')

# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_lasso3_ ' +
                        str(RMSE_by_base) + '.csv', index=False)
