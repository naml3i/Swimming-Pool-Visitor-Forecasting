
# Importing modules
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


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


# Baseline model 1: Predictions = Average number of visitors for the whole
# period in training data
predictions = np.mean(train['visitors_pool_total'])
test['visitors_pool_total'] = predictions


# Calculating error (RMSE) for the model
rmse_group = list()
for iter in xrange(CROSS_VALIDATION_ITER):
    train_validation, test_validation = create_validation_data(train)
    train_visitors_avg = np.mean(train_validation['visitors_pool_total'])
    test_validation['predict_visitors'] = train_visitors_avg
    test_validation['residual'] = test_validation['visitors_pool_total'] - \
        test_validation['predict_visitors']
    rmse = np.sqrt(np.mean(test_validation['residual'] *
                           test_validation['residual']))
    rmse_group.append(rmse)

RMSE = np.mean(rmse_group)
RMSE_by_base = RMSE / np.mean(train['visitors_pool_total'])


# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_baseline1_ ' +
                        str(RMSE_by_base) + '.csv', index=False)