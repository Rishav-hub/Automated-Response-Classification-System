from sentiment.entity.config_entity import PredictionPipelineConfig
from sentiment.exception import SentimentException
from sentiment.constant.training_pipeline import *
from sentiment.ml.model.lstm_model import SentimentLSTM
from sentiment.cloud_storage.s3_syncer import S3Sync

import sys
import numpy as np
from sentiment.logger import logging
import json
import torch

class PredictionPipeline:
    def __init__(self,):
        try:
            self.prediction_pipeline_config = PredictionPipelineConfig()
            self.s3_sync = S3Sync()
        except Exception as e:
            raise SentimentException(e, sys)

    def _delete_unwanted_symbols(self) -> dict:
        """Return the dictionary having ord() of all the symbols 

        Returns:
            dict: _description_
        """
        try:
            remove_dict = {ord(c):f'' for c in SYMBOLS_TO_DELETE}
            return remove_dict
        except Exception as e:
            raise SentimentException(e, sys)

    def _handle_punctuation(self, x):
        try:
            remove_dict = self._delete_unwanted_symbols()
            x = x.translate(remove_dict)
            return x
        except Exception as e:
            raise SentimentException(e, sys)


    def preprocess_string(self, x: str):
        """
        Applying all the preprocessing steps defined above.
        
        """
        try:
            x = x.lower()
            x = self._handle_punctuation(x)
            return x
        except Exception as e:
            raise SentimentException(e, sys)

    def padding_(self, sentences, seq_len):        
        try:
            features = np.zeros((len(sentences), seq_len),dtype=int)
            for ii, review in enumerate(sentences):
                if len(review) != 0:
                    features[ii, -len(review):] = np.array(review)[:seq_len]
            return features
        except Exception as e:
            raise SentimentException(e, sys)
    
    def get_vocab(self):
        try:
            with open(VOCAB_FILE_PATH, 'r') as fp:
                data = json.load(fp)
            return data
        except Exception as e:
            raise SentimentException(e, sys)
    
    def model_dict(self, vocab):
        try:
            model = SentimentLSTM(NO_LAYERS, vocab)
            return model
        except Exception as e:
            raise SentimentException(e, sys)

    def _get_model_in_production(self):
        try:
            s3_model_path = self.prediction_pipeline_config.s3_model_path
            model_download_path = self.prediction_pipeline_config.model_download_path
            os.makedirs(model_download_path, exist_ok=True)
            self.s3_sync.sync_folder_from_s3(folder=model_download_path, aws_bucket_url=s3_model_path)
            for file in os.listdir(model_download_path):
                if file.endswith(".pt"):
                    prediction_model_path = os.path.join(model_download_path, file)
                    logging.info(f"Production model for prediction found in {prediction_model_path}")
                    break
                else:
                    logging.info("Model is not available in Prediction artifacts")
                    prediction_model_path = None
            return prediction_model_path
        except Exception as e:
            raise SentimentException(e, sys)


    def predict_text(self, text: str):
        vocab = self.get_vocab()
        model = self.model_dict(len(vocab) + 1)
        word_seq = np.array([vocab[self.preprocess_string(word)] for word in text.split() 
                        if self.preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(self.padding_(word_seq,500))
        inputs = pad.to(DEVICE)
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        return(output.item())
        
    