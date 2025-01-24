import logging
import pickle
import sys
from typing import List, NamedTuple


import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


from kfp.v2 import compiler, dsl
from kfp.components import OutputPath, InputPath
from kfp.v2.dsl import Input, Output, Dataset, Model
import json

from google.cloud import aiplatform


logging.basicConfig(level="INFO", stream=sys.stdout)


def compile_pipeline(pipeline_func):   
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path="./temp/my_pipeline.json")

@dsl.component(packages_to_install=["ucimlrepo==0.0.7","fastparquet==2023.7.0"], base_image="python:3.9")
def load_data(dataset: Output[Dataset]):
    """
    Get iris dataset from UCI reposiory
    Returns: 
        X  - Dataframe containing 4 features regrding iris characteristics
        y - The target array for the iris classification 
    """
    import logging
    from ucimlrepo import fetch_ucirepo

    logging.info("Getting Dataset")
    data_iris = fetch_ucirepo(id=53) 
    X_array = data_iris.data.features 
    X_array.rename(columns = {
        'sepal length' : 'sepal_length',
        'sepal width' : 'sepal_width',
        'petal length' : 'petal_length',
        'petal width': 'petal_width'
    }, inplace=True)
    y_array = data_iris.data.targets['class']

    X_array['target'] = y_array

    X_array.to_csv(dataset.path)


@dsl.component(packages_to_install=["scikit-learn==1.5.2"], base_image="python:3.9")
def set_training_pipeline(pipeline_out: Output[Model])->Model:
    """
    Defines training pipeline steps and returns the pipeline
    """
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    pipeline = Pipeline(steps = [
        ('Imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
        ('normalization', StandardScaler()),
        ('estimator', LogisticRegression() )
        ]
    )
    joblib.dump(pipeline, pipeline_out.path)


@dsl.component(packages_to_install=["scikit-learn==1.5.2"], base_image="python:3.9")
def train_model(
    dataset: Input[Dataset],
    pipeline: Input[Model],
)->Model:
    import logging
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV, train_test_split

    dataset = pd.read_csv(dataset.path)
    logging.info(f"Spliting dataset")
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns="target"), dataset["target"], stratify=dataset["target"], test_size=0.2, random_state=14)

    logging.info(f"Fittig model with train data")


    parameters = {
        'estimator__solver': ['newton-cg'],
        'estimator__tol': [ 0.0001, 0.003, 0.01],
        'estimator__penalty': [None, 'l2'],
    }

    model = GridSearchCV(estimator=pipeline,
                            param_grid=parameters,
                            scoring= {"AUC": "roc_auc_ovr"},
                            refit="AUC",
                            cv=5,
                            verbose=1,
                            error_score='raise')
    
    pipeline = joblib.load(pipeline)
    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logging.info(f"Computing scores")
    model_score = model.score(X_test, y_test)
    logging.info(f"Model AUC Score: {model_score}")

    test_acc_score = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy test score: {test_acc_score}")

    logging.info("Saving model")
    model_artifact = Model(name="model", metadata={})
    joblib.dump(model, model_artifact.path)


PIPELINE_ROOT="/tmp"
@dsl.pipeline(
    name="my-pipeline-",
    description="A class project",
    pipeline_root=PIPELINE_ROOT
)
def my_pipeline_func():
    load_data_component = load_data()
    set_training_pipe_component = set_training_pipeline(
        
    ).after(load_data_component)

    fit_model_component = train_model(
        dataset = load_data_component.output,
        pipeline = set_training_pipe_component.outputs['pipeline_out']
    ).after(set_training_pipe_component)




def execute_pipeline():
    compile_pipeline(my_pipeline_func)
    PIPELINE_ROOT = "temp"
    aiplatform.init(project="personal-448814",
                    location="us-central1",
                    staging_bucket=f"gs://kfp_bucket_pipeline/{PIPELINE_ROOT}/")
    job = aiplatform.PipelineJob(
        display_name="A pipeline for a class project",
        template_path="./temp/my_pipeline.json",
        pipeline_root=f"gs://kfp_bucket_pipeline/{PIPELINE_ROOT}/",
        project="personal-448814",
        location="us-central1",
        enable_caching=False
        
    )
    job.run(
        service_account="personal@personal-448814.iam.gserviceaccount.com",
    )