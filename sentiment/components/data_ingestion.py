import os,sys
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sentiment.exception import SentimentException
from sentiment.logger import logging
from sentiment.entity.config_entity import DataIngestionConfig
from sentiment.entity.artifact_entity import DataIngestionArtifact
# from sentiment.data_access.sentiment_data import SentimentData
from sentiment.utils.main_utils import read_yaml_file
from sentiment.constant.training_pipeline import SCHEMA_FILE_PATH
#from sentiment.constant.s3_bucket import TRAINING_BUCKET_NAME
from sentiment.configuration.s3_operations import S3Operation


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig, s3_operations: S3Operation):
        try:
            self.data_ingestion_config=data_ingestion_config
            self.s3_operations = s3_operations
        except Exception as e:
            raise SentimentException(e,sys)


    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.raw_data_dir, exist_ok=True)
            self.s3_operations.read_data_from_s3(self.data_ingestion_config.s3_file_name,self.data_ingestion_config.bucket_name,
                                                self.data_ingestion_config.local_file_path)
            logging.info("Exited the get_data_from_s3 method of Data ingestion class")
        except Exception as e:
            raise SentimentException(e, sys) from e        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.get_data_from_s3()
            data_ingestion_artifacts = DataIngestionArtifact(ingested_file_path= self.data_ingestion_config.local_file_path,
                            ingested_file_name= self.data_ingestion_config.s3_file_name)
            return data_ingestion_artifacts
        except Exception as e:
            raise SentimentException(e,sys)