import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


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


# Creating linear regression object and predictors
model = linear_model.LinearRegression(fit_intercept=True, normalize=True)
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

    # print
    # print 'Model #', str(model_counter)
    # print 'Coefficients: \n', coefficients
    rmse = np.sqrt(np.mean((model.predict(test_validation[predictors])
                            - test_validation['visitors_pool_total']) ** 2))
    # print 'RMSE: %.2f' % rmse
    # print 'R-sqr: %.4f' % model.score(test_validation[predictors],
    #                                  test_validation['visitors_pool_total'])
    rmse_group.append(rmse)

RMSE = np.mean(rmse_group)
RMSE_by_base = RMSE / np.mean(train['visitors_pool_total'])
# print 'Mean RMSE: %.4f' % RMSE_by_base

# Model 4: Predictions = Linear regression using variables from main training
# file (excluding weather data) [SCORE: 644.45123]
model.fit(train[predictors], train['visitors_pool_total'])
test['visitors_pool_total'] = model.predict(test[predictors])


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
plt.savefig('../Plots/linear_regression1_actual_vs_predictions.png')

# Creating a submission file
submission_model = test[['date', 'visitors_pool_total']]
submission_model.to_csv('../Submissions/submission_linear_regression1_ ' +
                        str(RMSE_by_base) + '.csv', index=False)
