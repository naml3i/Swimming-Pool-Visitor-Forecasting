import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


CROSS_VALIDATION_ITER = 10


# Reading training, test, weather data and submission file
train = pd.read_csv('../Data/nettebad_train_set.csv')
test = pd.read_csv('../Data/nettebad_test_set.csv')
weather_dwd_train = pd.read_csv('../Data/weather_dwd_train_set.csv')
weather_dwd_test = pd.read_csv('../Data/weather_dwd_test_set.csv')
weather_uni_train = pd.read_csv('../Data/weather_uni_osnabrueck_train_set.csv')
weather_uni_test = pd.read_csv('../Data/weather_uni_osnabrueck_test_set.csv')
submission = pd.read_csv('../Submissions/sample_submission_nettebad.csv')

# print train.head()
# print test.head()
# print submission.head()


# Correcting date format for 'weather_uni_train.csv'
for row in xrange(weather_uni_train.shape[0]):
    dt = datetime.strptime(weather_uni_train.loc[row, 'date'], '%Y-%m-%d')
    weather_uni_train.loc[row, 'year'] = dt.year
    weather_uni_train.loc[row, 'month'] = dt.month
    weather_uni_train.loc[row, 'day'] = dt.day

# Merge data sets
train = train.merge(weather_dwd_train, how='left', on='date')

for row in xrange(train.shape[0]):
    dt = datetime.strptime(train.loc[row, 'date'], '%m/%d/%Y')
    train.loc[row, 'year'] = dt.year
    train.loc[row, 'month'] = dt.month
    train.loc[row, 'day'] = dt.day

train = train.merge(weather_uni_train, how='left', on=['year', 'month', 'day'])
test = test.merge(weather_dwd_test, how='left', on='date')
test = test.merge(weather_uni_test, how='left', on='date')

# train.to_csv('../Data/train_modified.csv', index=False)
# test.to_csv('../Data/test_modified.csv', index=False)


# Importing modified datasets & removing NAs
train = pd.read_csv('../Data/train_modified.csv')
test = pd.read_csv('../Data/test_modified.csv')
train.drop(['year', 'month', 'day', 'date_y'], axis=1, inplace=True)
train.rename(columns={'date_x': 'date'}, inplace=True)
train.dropna(inplace=True)
train = pd.concat([train, pd.get_dummies(
    train[['wind_direction_category_UniOS']],
    prefix=['wind_direction'])], axis=1)
train.drop(['wind_direction_category_UniOS'], axis=1, inplace=True)
test = pd.concat([test, pd.get_dummies(test[['wind_direction_category_UniOS']],
                                       prefix=['wind_direction'])], axis=1)
test.drop(['wind_direction_category_UniOS'], axis=1, inplace=True)
# train.to_csv('../Data/train_modified2.csv', index=False)
# test.to_csv('../Data/test_modified2.csv', index=False)


# Removing columns with zero variance
train.drop(['freizeitbad_closed', 'sauna_closed', 'kursbecken_closed',
            'price_adult_max', 'price_reduced_max', 'sloop_dummy'],
           axis=1, inplace=True)
test = test.loc[:, (test != test.ix[0]).any()]

# Extract validation data from training data
def create_validation_data(training_data):
    train_validation, test_validation = train_test_split(training_data,
                                                         test_size=0.3)
    return train_validation, test_validation


# Creating linear regression object and predictors
model = linear_model.LinearRegression(fit_intercept=True, normalize=True)
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

# Model 7: Predictions = Linear regression using variables from main training
# file plus derived variable - 'day of the week' plus weather data
# (excluding missing data)
# [SCORE: 445.32036]
train['day_of_week'] = pd.DatetimeIndex(train['date']).weekday
test['day_of_week'] = pd.DatetimeIndex(test['date']).weekday
model.fit(train[predictors], train['visitors_pool_total'])
test['visitors_pool_total'] = model.predict(test[predictors])


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
plt.savefig('../Plots/linear_regression4_actual_vs_predictions.png')

# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
for row in xrange(submission_model.shape[0]):
    if submission_model.loc[row, 'visitors_pool_total'] > 5000:
        submission_model.loc[row, 'visitors_pool_total'] = int(np.mean(
            train['visitors_pool_total']))
submission_model.to_csv('../Submissions/submission_linear_regression4_ ' +
                        str(RMSE_by_base) + '.csv', index=False)
