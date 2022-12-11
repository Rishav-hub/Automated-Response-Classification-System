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
from sentiment.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact
from sentiment.utils.main_utils import read_yaml_file,\
            load_numpy_array_data
from sentiment.constant.training_pipeline import SCHEMA_FILE_PATH, DEVICE, NO_LAYERS
from sentiment.configuration.s3_operations import S3Operation
from sentiment.ml.model.lstm_model import SentimentLSTM
from sentiment.ml.metric.classification_metric import acc
from sentiment.cloud_storage.s3_syncer import S3Sync
from sentiment.components.model_trainer import ModelTrainer


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            self.model_evaluation_config=model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SentimentException(e,sys)
    
    def get_best_model_path(self):
        try:
            model_evaluation_artifacts_dir = self.model_evaluation_config.model_evaluation_artifacts_dir
            os.makedirs(model_evaluation_artifacts_dir, exist_ok=True)
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pt"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info("Model is not available in best_model_directory")
            return best_model_path
        except Exception as e:
            raise SentimentException(e,sys)

    def evaluate_model(self):
        best_model_path = self.get_best_model_path()
        if best_model_path is not None:
            model = SentimentLSTM(NO_LAYERS, self.model_trainer_artifact.vocab)
            # load back the model
            state_dict = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            accuracy = state_dict['accuracy']
            loss = state_dict['loss']
            logging.info(f"S3 Model Validation accuracy is {accuracy}")
            logging.info(f"S3 Model Validation loss is {loss}")
            s3_model_accuracy = accuracy
            s3_model_loss = loss
        else:
            logging.info("Model is not found on production server, So couldn't evaluate")
            s3_model_accuracy = None
            s3_model_loss = None
        return s3_model_accuracy, s3_model_loss
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            s3_model_accuracy, s3_model_loss = self.evaluate_model()
            tmp_best_model_accuracy = 0 if s3_model_accuracy is None else s3_model_accuracy
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss
            trained_model_accuracy = self.model_trainer_artifact.trained_model_accuracy
            trained_model_loss = self.model_trainer_artifact.trained_model_loss
            evaluation_response = trained_model_accuracy > tmp_best_model_accuracy and tmp_best_model_loss > trained_model_loss
            model_evaluation_artifacts = ModelEvaluationArtifact(
                trained_model_accuracy = trained_model_accuracy,
                s3_model_accuracy = s3_model_accuracy,
                is_model_accepted = evaluation_response,
                trained_model_path = self.model_trainer_artifact.trained_model_dir_path,
                s3_model_path = self.model_evaluation_config.s3_model_path
            )
            return model_evaluation_artifacts
        except Exception as e:
            raise SentimentException(e,sys)