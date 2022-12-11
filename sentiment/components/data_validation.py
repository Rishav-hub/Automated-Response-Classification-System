import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import shutil


from sentiment.exception import SentimentException
from sentiment.logger import logging
from sentiment.entity.config_entity import DataValidationConfig
from sentiment.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
# from sentiment.data_access.sentiment_data import SentimentData
from sentiment.utils.main_utils import read_yaml_file
from sentiment.constant.training_pipeline import SCHEMA_FILE_PATH
#from sentiment.constant.s3_bucket import TRAINING_BUCKET_NAME
from sentiment.configuration.s3_operations import S3Operation


class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifacts: DataIngestionArtifact):
        try:
            logging.info("Staring Data Validation >>>>>")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifacts= data_ingestion_artifacts
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SentimentException(e,sys)

    def _accepted_columns(self):
        try:
            col_names_list = []
            column_names: list = self._schema_config['columns']
            for names in column_names:
                col_names_list.append(list(names.keys())[0])
            return col_names_list
        except Exception as e:
            raise SentimentException(e, sys)

    def read_data(self) -> pd.DataFrame:
        try:
            ingested_data = pd.read_csv(self.data_ingestion_artifacts.ingested_file_path,\
                                        encoding=self.data_validation_config.encoding,
                                        names= self._accepted_columns())
            return ingested_data
        except Exception as e:
            raise SentimentException(e,sys)

    def check_column_type(self) -> bool:
        col_dtype_list = []
        column_dtype: list = self._schema_config['columns']
        for names in column_dtype:
            col_dtype_list.append(list(names.values())[0])
        
        for r in range(len(col_dtype_list)):
            if col_dtype_list[r] != self.read_data().dtypes.to_list()[r]:
                return False
        return True
    
    @staticmethod
    def save_file(previous_path: str, new_path: str) -> None:
        try:
            shutil.copy(previous_path, new_path)
        except shutil.SameFileError:
            raise (" Source and Destination is Same")
        except PermissionError:
            raise ('Permission Denied')
        except Exception as e:
            raise SentimentException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            if self.check_column_type():
                logging.info("Data Accepted>>>>")
                ingested_data = self.read_data()
                validation_status = True
                # Copy the file from ingested artifacts to Validated artifacts
                os.makedirs(self.data_validation_config.validated_data_dir, exist_ok= True)
                ingested_data.to_csv(self.data_validation_config.validated_file_name, index= False)

            else:
                validation_status= False
                logging.info("Data Not Accepted>>>>")
            data_validation_artifacts= DataValidationArtifact(validation_status=validation_status,\
                                            valid_file_path= self.data_validation_config.validated_file_name)
            return data_validation_artifacts
        except Exception as e:
            raise SentimentException(e,sys)

    
if __name__ == "__main__":
    dataval = DataValidation(data_validation_config = DataValidationConfig(), 
            data_ingestion_artifacts= DataIngestionArtifact())
