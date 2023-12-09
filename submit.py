import problem
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()

pipeline = problem.get_estimator()

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

X_eval = problem.get_eval_data()

y_pred = pipeline.predict(X_eval)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)

X_transformed = problem._encode_data(X_eval)

comparison_date = pd.to_datetime('2021-09-15')
for i in range(len(X_eval)):
    if X_transformed['counter_id'][i] == 0 or X_transformed['counter_id'][i] == 1:
        if  X_eval['date'][i] >= comparison_date:
            results['log_bike_count'][i] = 0

results.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")