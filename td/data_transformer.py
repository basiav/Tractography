from enum import Enum
from pipeline_encoder import EncodingPipeline
import pandas as pd
import numpy as np


class DataTransformer:
    class FeatureMethod(Enum):
        COSINE = "Cos"
        RBF = "RBF"
        PCA = "PCA"
        NMF = "NMF"

    COMPONENT_PREFIX = "Comp"
    COLUMNS_TO_DROP = ["volume", "site", "OS", "ID", "ltdi", "tdi"]
    NUMERICAL_COLUMNS_TO_DROP = ["OS (corrected)", "status"]

    @staticmethod
    def add_extracted_mapsdata(df_base: pd.DataFrame, extracted_features: np.ndarray, method: FeatureMethod, n_components: int, column_names: list[str] | None = None) -> pd.DataFrame:
        if column_names is None:
            column_names = [f"{method.value}_{DataTransformer.COMPONENT_PREFIX}_{i+1}" for i in range(extracted_features.shape[1])]

        print(extracted_features.shape)
        print(len(column_names))
        
        df_new_features = pd.DataFrame(
            extracted_features,
            columns=column_names, 
            index=df_base.index
        )

        df_all = pd.concat([df_base, df_new_features], axis=1)

        return df_all
    
    @staticmethod
    def drop_columns(df_base: pd.DataFrame):
        columns_to_drop = DataTransformer.COLUMNS_TO_DROP
        df_base = df_base.drop(columns=columns_to_drop)

        return df_base
    
    @staticmethod
    def feature_fit_transform(encoding_pipeline: EncodingPipeline):
        df_base = encoding_pipeline.df
        
        categorical_features = df_base.select_dtypes(include="object").columns
        numerical_features = df_base.select_dtypes(exclude="object").columns

        numerical_features = numerical_features.drop(DataTransformer.NUMERICAL_COLUMNS_TO_DROP)

        X_transformed = encoding_pipeline.fit_transform(list(numerical_features), list(categorical_features))

        return X_transformed
    

    @staticmethod
    def feature_transform(test_df: pd.DataFrame, encoding_pipeline: EncodingPipeline):
        X_transformed = encoding_pipeline.transform(test_df)
        
        return X_transformed