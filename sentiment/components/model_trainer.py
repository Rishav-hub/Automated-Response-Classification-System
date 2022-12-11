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
from sentiment.entity.config_entity import ModelTrainerConfig
from sentiment.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sentiment.utils.main_utils import read_yaml_file,\
            load_numpy_array_data
from sentiment.constant.training_pipeline import SCHEMA_FILE_PATH, DEVICE, S3_BUCKET_MODEL_URI
from sentiment.configuration.s3_operations import S3Operation
from sentiment.ml.model.lstm_model import SentimentLSTM
from sentiment.ml.metric.classification_metric import acc
from sentiment.cloud_storage.s3_syncer import S3Sync



class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact) -> None:
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SentimentException(e,sys)
    
    def _get_numpy_data(self):
        try:
            x_train_pad = load_numpy_array_data(self.data_transformation_artifact.train_data_file_path)
            x_test_pad = load_numpy_array_data(self.data_transformation_artifact.test_data_file_path)
            y_train = load_numpy_array_data(self.data_transformation_artifact.train_labels_file_path)
            y_test = load_numpy_array_data(self.data_transformation_artifact.test_labels_file_path)

            return x_train_pad, x_test_pad, y_train, y_test
        except Exception as e:
            raise SentimentException(e, sys)

    def get_tensor_dataset(self,x_train_pad: np.array, x_test_pad: np.array, y_train: np.array, y_test: np.array):
        try:
            train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
            valid_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))
            return train_data, valid_data
        except Exception as e:
            raise SentimentException(e, sys)


    def get_data_loader(self,train_data, valid_data):
        try:
            train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=self.model_trainer_config.batch_size)
            valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=self.model_trainer_config.batch_size)
            return train_loader, valid_loader
        except Exception as e:
            raise SentimentException(e, sys)

    def initiate_model(self):
        try:
            model = SentimentLSTM(self.model_trainer_config.no_layers,len(self.data_transformation_artifact.vocab) + 1)
            return model
        except Exception as e:
            raise SentimentException(e, sys)

    def training_loop(self, model: object, train_loader: object, valid_loader: object):
        try:
            epoch_tr_loss,epoch_vl_loss = [],[]
            epoch_tr_acc,epoch_vl_acc = [],[]

            valid_loss_min = np.Inf

            criterion = nn.BCELoss()

            optimizer = optim.Adam(model.parameters(), lr=self.model_trainer_config.learning_rate)

            for epoch in range(self.model_trainer_config.epochs):
                train_losses = []
                train_acc = 0.0
                logging.info("Starting with the training..............")
                model.train()
                
                logging.info("Initializing Hidden state...............")
                # initialize hidden state 
                h = model.init_hidden(self.model_trainer_config.batch_size)
                logging.info("Initializing Input to device from Train Loader...............")

                for inputs, labels in train_loader:
                    
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)   
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    h = tuple([each.data for each in h])
                    
                    model.zero_grad()
                    output,h = model(inputs,h)
                    
                    # calculate the loss and perform backprop
                    loss = criterion(output.squeeze(), labels.float())
                    loss.backward()
                    train_losses.append(loss.item())
                    # calculating accuracy
                    accuracy = acc(output,labels)
                    train_acc += accuracy
                    #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    nn.utils.clip_grad_norm_(model.parameters(), self.model_trainer_config.clip)
                    optimizer.step()
            
                
                    
                val_h = model.init_hidden(self.model_trainer_config.batch_size)
                val_losses = []
                val_acc = 0.0
                model.eval()
                logging.info("Initializing Input to device from Valid Loader...............")

                for inputs, labels in valid_loader:
                        
                    val_h = tuple([each.data for each in val_h])

                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                    accuracy = acc(output,labels)
                    val_acc += accuracy
                        
                epoch_train_loss = np.mean(train_losses)
                epoch_val_loss = np.mean(val_losses)
                epoch_train_acc = train_acc/len(train_loader.dataset)
                epoch_val_acc = val_acc/len(valid_loader.dataset)
                epoch_tr_loss.append(epoch_train_loss)
                epoch_vl_loss.append(epoch_val_loss)
                epoch_tr_acc.append(epoch_train_acc)
                epoch_vl_acc.append(epoch_val_acc)
                logging.info(f'Epoch {epoch+1}') 
                logging.info(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
                logging.info(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
                if epoch_val_loss <= valid_loss_min:
                    os.makedirs(self.model_trainer_config.model_trainer_artifacts_dir, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "accuracy": epoch_val_acc,
                        "loss": epoch_val_loss}, self.model_trainer_config.trained_model_dir)
                    logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
                    valid_loss_min = epoch_val_loss
                logging.info(25*'==')
            return epoch_val_loss, epoch_val_acc
        except Exception as e:
            raise SentimentException(e, sys)

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        try:
            logging.info("Getting Data from previous stage >>>>>>>>")
            x_train_pad, x_test_pad, y_train, y_test = self._get_numpy_data()
            logging.info("Get torch dataset >>>>>>>>>")
            train_data, valid_data = self.get_tensor_dataset(x_train_pad, x_test_pad, y_train, y_test)
            logging.info("Get torch Data Loader object >>>>>>>>>")
            train_loader, valid_loader = self.get_data_loader(train_data, valid_data)
            logging.info("Initiate Model Object >>>>>>>")
            model = self.initiate_model()
            logging.info("Starting Model Training Loop >>>>>>")
            epoch_val_loss, epoch_val_acc = self.training_loop(model, train_loader, valid_loader)
            s3_sync = S3Sync()
            logging.info("Sync transformation files to S3 transformation artifacts folder...") 
            s3_sync.sync_folder_to_s3(folder=self.model_trainer_config.model_trainer_artifacts_dir, aws_bucket_url=S3_BUCKET_MODEL_URI)
            logging.info("Finished Syncing files to S3 transformation artifacts folder")
            model_trainer_artifacts =  ModelTrainerArtifact(trained_model_dir_path = self.model_trainer_config.trained_model_dir,
                            trained_model_accuracy = epoch_val_acc,
                            trained_model_loss =  epoch_val_loss,
                            vocab = len(self.data_transformation_artifact.vocab) + 1  )
            return model_trainer_artifacts

        except Exception as e:
            raise SentimentException(e, sys)