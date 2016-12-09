import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


CROSS_VALIDATION_ITER = 10

# Reading training, test data and submission file
train = pd.read_csv('../Data/nettebad_train_set.csv')
test = pd.read_csv('../Data/nettebad_test_set.csv')
submission = pd.read_csv('../Submissions/sample_submission_nettebad.csv')

# print train.head()
# print test.head()
# print submission.head()


# Extract validation data from training data
def create_validation_data(training_data):
    train_validation, test_validation = train_test_split(training_data,
                                                         test_size=0.2)
    return train_validation, test_validation


# Calculating error (RMSE) for the model
rmse_group = list()
for iter in xrange(CROSS_VALIDATION_ITER):
    train_validation, test_validation = create_validation_data(train)
    train_validation = train_validation.reset_index(drop=True)
    test_validation = test_validation.reset_index(drop=True)
    train_validation['day_of_week'] = pd.DatetimeIndex(
        train_validation['date']).weekday
    for row in xrange(train_validation.shape[0]):
        train_validation.loc[row, 'weekend'] = 1 \
            if train_validation.loc[row, 'day_of_week'] >= 5 else 0
    train_validation_combined = train_validation[
        ['visitors_pool_total', 'weekend']]
    train_validation_combined = train_validation_combined.groupby(
        ['weekend']).mean().reset_index()
    train_validation_dict = train_validation_combined.to_dict()

    test_validation['day_of_week'] = pd.DatetimeIndex(
        test_validation['date']).weekday
    for row in xrange(test_validation.shape[0]):
        test_validation.loc[row, 'weekend'] = 1 \
            if test_validation.loc[row, 'day_of_week'] >= 5 else 0

    for row in xrange(test_validation.shape[0]):
        test_validation.loc[row, 'predict_visitors'] = \
            train_validation_dict['visitors_pool_total'][0] \
                if test_validation.loc[row, 'weekend'] == 0 \
                else train_validation_dict['visitors_pool_total'][1]

    test_validation['residual'] = test_validation['visitors_pool_total'] - \
        test_validation['predict_visitors']
    rmse = np.sqrt(np.mean(test_validation['residual'] *
                           test_validation['residual']))
    rmse_group.append(rmse)

RMSE = np.mean(rmse_group)
RMSE_by_base = RMSE / np.mean(train['visitors_pool_total'])


# Baseline model 2: Predictions = Average number of visitors based on weekday
# or weekend [SCORE: 526.31883]
train['day_of_week'] = pd.DatetimeIndex(train['date']).weekday
for row in xrange(train.shape[0]):
    train.loc[row, 'weekend'] = 1 if train.loc[row, 'day_of_week'] >= 5 else 0
train_combined = train[['visitors_pool_total', 'weekend']]
train_combined = train_combined.groupby(['weekend']).mean().reset_index()
train_dict = train_combined.to_dict()

test['day_of_week'] = pd.DatetimeIndex(test['date']).weekday
for row in xrange(test.shape[0]):
    test.loc[row, 'weekend'] = 1 if test.loc[row, 'day_of_week'] >= 5 else 0

for row in xrange(test.shape[0]):
    test.loc[row, 'visitors_pool_total'] = train_dict[
        'visitors_pool_total'][0] if test.loc[row, 'weekend'] == 0 \
        else train_dict['visitors_pool_total'][1]


# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_baseline2_ ' +
                        str(RMSE_by_base) + '.csv', index=False)