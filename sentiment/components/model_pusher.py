import os,sys
from collections import  Counter

from sklearn.model_selection import train_test_split

import numpy as np
from pandas import DataFrame
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sentiment.exception import SentimentException
from sentiment.logger import logging
from sentiment.entity.config_entity import ModelEvaluationConfig
from sentiment.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from sentiment.utils.main_utils import read_yaml_file,\
            load_numpy_array_data
from sentiment.constant.training_pipeline import TRAINED_MODEL_NAME
from sentiment.configuration.s3_operations import S3Operation
from sentiment.ml.model.lstm_model import SentimentLSTM
from sentiment.ml.metric.classification_metric import acc
from sentiment.cloud_storage.s3_syncer import S3Sync
from sentiment.components.model_trainer import ModelTrainer


class ModelPusher:
    def __init__(self, model_evaluation_artifacts: ModelEvaluationArtifact):
        self.model_evaluation_artifacts = model_evaluation_artifacts
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Initiating model pusher component")
            if self.model_evaluation_artifacts.is_model_accepted:
                trained_model_path = self.model_evaluation_artifacts.trained_model_path
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                s3_sync = S3Sync()
                s3_sync.sync_folder_to_s3(folder=trained_model_path, aws_bucket_url=s3_model_folder_path)
                message = "Model Pusher pushed the current Trained model to Production server storage"
                response = {"is model pushed": True, "S3_model": s3_model_folder_path + "/" + str(TRAINED_MODEL_NAME),"message" : message}
                logging.info(response)
                
            else:
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                message = "Current Trained Model is not accepted as model in Production has better accuracy"
                response = {"is model pushed": False, "S3_model":s3_model_folder_path,"message" : message}
                logging.info(response)
            
            model_pusher_artifacts = ModelPusherArtifact(response=response)
            return model_pusher_artifacts
        except Exception as e:
            raise SentimentException(e, sys)