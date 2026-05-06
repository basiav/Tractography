from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
)
from feature_engine.encoding import (
    RareLabelEncoder,
)
import sklearn
import pandas as pd


class EncodingPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fit_transform(self, numerical_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
        sklearn.set_config(transform_output="pandas")

        one_hot_encoder = OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )
        rle = RareLabelEncoder(n_categories=1, tol=0.01, missing_values='ignore')

        categorical_pipeline = Pipeline(
            [
                ("rare_label_encoder", rle),
                ("one_hot_encoder", one_hot_encoder),
            ],
        )

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])

        self.full_preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
            remainder='passthrough',
            verbose_feature_names_out=False,)

        X = self.full_preprocessor.fit_transform(self.df)
        return X
    
    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
            if self.full_preprocessor is None:
                raise RuntimeError("Najpierw wywołaj fit_transform na danych treningowych!")
            
            # Używamy TYLKO .transform(), żeby zachować skale i kategorie z UCSF
            return self.full_preprocessor.transform(test_df)