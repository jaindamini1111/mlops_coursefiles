import pandas as pd
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pickle

@task
def get_paths(date):
    if date == None:
        date_pivot = datetime.now()
    else:
        date = datetime.strptime(date, "%Y-%m-%d")
        date_pivot = date
    
    date_train = date_pivot - pd.DateOffset(months = 2)
    date_val = date_pivot - pd.DateOffset(months = 1)    
    train_path = "./data/fhv_tripdata_{:04d}-{:02d}.parquet".format(date_train.year, date_train.month)
    val_path = "./data/fhv_tripdata_{:04d}-{:02d}.parquet".format(date_val.year, date_val.month)
    
    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info("The mean duration of training is: {0}".format(mean_duration))
#         print(f"The mean duration of training is {mean_duration}")
    else:
        logger.info("The mean duration of validation is: {0}".format(mean_duration))
#         print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values
    
    #     print(f"The shape of X_train is {X_train.shape}")
    logger.info("The shape of X_train is: {0}".format(X_train.shape))
#     print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info("The DictVectorizer has {0} features".format(len(dv.feature_names_)))

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    #     print(f"The MSE of training is: {mse}")
    logger.info("The MSE of training is: {0}".format(mse))
    
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    #     print(f"The MSE of validation is: {mse}")
    logger.info("The MSE of validation is: {0}".format(mse))
    
    return

@flow(task_runner=SequentialTaskRunner())
def main(date):
    train_path, val_path = get_paths(date).result()
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    # model and artifacts
    with open("artifacts/model-{0}.bin".format(date), "wb") as f_out:
        pickle.dump(lr, f_out)
    with open("artifacts/enc-{0}.bin".format(date), "wb") as f_out:
        pickle.dump(dv, f_out)


# main(date = "2021-08-15")

DeploymentSpec(flow = main, name = "model_mlopszoomcamp_week03", schedule = CronSchedule(cron = "0 9 15 * *", timezone = "America/New_York"),  flow_runner = SubprocessFlowRunner(), tags = ["mlopszoomcamp_w3"])