from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import numpy as np
from scipy.special import softmax
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(model, X_test, y_test):
    print("X_test shape:", X_test.shape)
    predictions = model.predict(X_test)
    print_regression_metrics(predictions, y_test)


def evaluate_XGB(obj, X_test, y_test):
    dtest = xgb.DMatrix(data=X_test)
    predictions = obj.predict(dtest)
    print_regression_metrics(predictions, y_test)


def print_regression_metrics(predictions, y_test):
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
