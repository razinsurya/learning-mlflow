import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__=="__main__":
    
    # Data Ingestion - Reading the dataset (wine quality dataset)
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download the data")
        sys.exit(1)

    # Split the data into train and test
    train, test = train_test_split(data, random_state=42)
    
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        
        print("ElasticNet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: {}".format(rmse))
        print("  MAE: {}".format(mae))
        print("  R2:  {}".format(r2))
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # For the remote server (AWS, etc.) we need the correct tracking URI
        remote_server_uri = "http://127.0.0.1:5000/"
        mlflow.set_tracking_uri(remote_server_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Model logging depending on whether the tracking URI is local or remote
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name="ElasticNetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
