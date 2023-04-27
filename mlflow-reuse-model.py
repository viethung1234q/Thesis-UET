import os
import mlflow
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


os.environ["AWS_ACCESS_KEY_ID"] = "hungdv"
os.environ["AWS_SECRET_ACCESS_KEY"] = "hungdv123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://10.10.6.141:9190" 

mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000") 
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

def train(file_path) -> str:
    logged_model = 'runs:/8bf1e39afc1e490ebb6f0740d755dd8f/model_mlp' # runs:/<mlflow_run_id>/run-relative/path/to/model
    
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    
    df = pd.read_csv(file_path)

    labels_column = "income"
    train_x, test_x, train_y, test_y = train_test_split(
        df.drop([labels_column], axis=1),
        df[labels_column],
        random_state=69
    )
    
    mlflow.set_experiment(experiment_name="Income")
    with mlflow.start_run(run_name="income_training"):
#         alpha, hidden_layers = 1e-3, (6, 4)
#         mlp = MLPClassifier(
#             solver="lbfgs",
#             alpha=alpha,
#             hidden_layer_sizes=hidden_layers,
#             random_state=69,
#         )

#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("hidden_layers", hidden_layers)

        # mlp.fit(train_x, train_y)

        # preds = mlp.predict(test_x)
        
        # Predict on a Pandas DataFrame.
        preds = loaded_model.predict(pd.DataFrame(test_x))

        accuracy = (test_y == preds).sum() / preds.shape[0]
        mlflow.log_metric("accuracy", accuracy)

        # result = mlflow.sklearn.log_model(
        #     sk_model=mlp,
        #     artifact_path="model",
        #     registered_model_name="income_model",
        # )
        # return f"{mlflow.get_artifact_uri()}/{result.artifact_path}"


if __name__ == '__main__':
    # preprocess("data/adult.csv", "data/output.csv")
    train("data/output.csv")
