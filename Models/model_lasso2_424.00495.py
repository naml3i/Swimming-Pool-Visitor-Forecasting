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


# Creating lasso regression object (alpha = 0.3, max_iter=10000) and predictors
model = linear_model.Lasso(alpha=0.3, fit_intercept=True, normalize=True,
                           max_iter=10000)
predictors = list(train.columns.values)
predictors.remove('date')
predictors.remove('visitors_pool_total')
predictors.append('day_of_week')

# Calculating error (RMSE) for the model
rmse_group = list()
model_counter = 0
for iter in xrange(CROSS_VALIDATION_ITER):
    model_counter += 1
    train_validation, test_validation = create_validation_data(train)
    train_validation['day_of_week'] = pd.DatetimeIndex(
        train_validation['date']).weekday
    test_validation['day_of_week'] = pd.DatetimeIndex(
        test_validation['date']).weekday
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

# Model 9: Predictions = Lasso regression (with tuned params) using sparse
# training set plus derived variable - 'day of the week' plus weather data
# (excluding missing data)
# [SCORE: 424.00495]
train['day_of_week'] = pd.DatetimeIndex(train['date']).weekday
test['day_of_week'] = pd.DatetimeIndex(test['date']).weekday
model.fit(train[predictors], train['visitors_pool_total'])
test['visitors_pool_total'] = model.predict(test[predictors])
test['visitors_pool_total'] = test['visitors_pool_total'].astype(int)


# Plot cross-validated predictions
fig, ax = plt.subplots()
train_validation, test_validation = create_validation_data(train)
train_validation['day_of_week'] = pd.DatetimeIndex(train_validation['date']) \
    .weekday
test_validation['day_of_week'] = pd.DatetimeIndex(test_validation['date']) \
    .weekday
y = train_validation['visitors_pool_total']
train_validation.drop(['date', 'visitors_pool_total'], axis=1, inplace=True)
predicted = cross_val_predict(model, train_validation, y, cv=10)
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.savefig('../Plots/lasso2_actual_vs_predictions.png')

# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_lasso2_ ' +
                        str(RMSE_by_base) + '.csv', index=False)
