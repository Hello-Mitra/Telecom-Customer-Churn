from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from src.features.build_features import get_preprocessor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns


def tune_logistic_regression_optuna(df):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Telco Customer Churn - Logistic Regression HP Optuna")

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):

        C = trial.suggest_float("C", 1e-4, 10, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=23
        )

        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        mean_score = scores.mean()

        with mlflow.start_run(nested=True):
            mlflow.log_param("C", C)
            mlflow.log_param("penalty", penalty)
            mlflow.log_param("solver", solver)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_metric("cv_auc", mean_score)

        return mean_score


    with mlflow.start_run(run_name="LogisticRegression_Optuna_Tuning") as run:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best_params = study.best_params

        mlflow.log_params(best_params)

        best_model = LogisticRegression(random_state=42, **best_params)

        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("model", best_model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_prob = pipeline.predict_proba(X_test)[:,1]
        test_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("best_cv_auc", study.best_value)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

        # Classification Report
        report_text = classification_report(y_test, y_pred)

        with open("classification_report.txt","w") as f:
            f.write(report_text)

        mlflow.log_artifact("classification_report.txt")
        mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="best_model"
                )

        print("Best Params:", best_params)

    run_id = run.info.run_id
    return study, run_id
