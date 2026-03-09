import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer


def get_preprocessor() -> ColumnTransformer:

    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """

    numeric_feat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    categorical_feat_pipeline = Pipeline(steps=[
        ('ohe_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_feat_pipeline, make_column_selector(dtype_include=['int64','float64'])),
            ('cat', categorical_feat_pipeline, make_column_selector(dtype_include=['object','category']))
        ],
        remainder='drop'
    )

    return preprocessor