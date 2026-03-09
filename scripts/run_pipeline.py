from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train import tune_logistic_regression_optuna


def run_pipeline():

    df = load_data("data/raw/Telco-Customer-Churn.csv")

    df = preprocess_data(df)

    study, run_id = tune_logistic_regression_optuna(df)

    print("Pipeline finished successfully")


if __name__ == "__main__":
    run_pipeline()