import os,sys
from collections import  Counter
from typing import Optional
import json

from sklearn.model_selection import train_test_split

import numpy as np
from pandas import DataFrame
import pandas as pd

from sentiment.exception import SentimentException
from sentiment.logger import logging
from sentiment.entity.config_entity import DataTransformationConfig
from sentiment.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from sentiment.utils.main_utils import read_yaml_file, save_numpy_array_data
from sentiment.constant.training_pipeline import SCHEMA_FILE_PATH, STOPWORD, CORPUS,\
            S3_ARTIFACTS_URI, ENCODING, VOCAB_FILE_PATH
from sentiment.configuration.s3_operations import S3Operation
from sentiment.cloud_storage.s3_syncer import S3Sync

#nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.util import ngrams
stop=set(stopwords.words(STOPWORD))
nltk.download(CORPUS)

class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifacts: DataValidationArtifact) -> None:
        try:
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifacts = data_validation_artifacts
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SentimentException(e,sys)
    
    def change_target_values(self, raw_data_path: str) -> DataFrame:
        """Change the initial target values of 4 to 2

        Returns:
            DataFrame: transformed dataframe
        """
        t = []
        raw_data = pd.read_csv(raw_data_path, encoding=ENCODING)
        for i in raw_data.target.values:
            if i ==0: t.append(0)
            else: t.append(1)
                
        selected_df = pd.DataFrame({"text":raw_data.text.values, "target":t})
        return selected_df
    
    def _delete_unwanted_symbols(self) -> dict:
        """Return the dictionary having ord() of all the symbols 

        Returns:
            dict: _description_
        """
        remove_dict = {ord(c):f'' for c in self.data_transformation_config.symbols_to_delete}
        return remove_dict

    def _handle_punctuation(self, x):
        remove_dict = self._delete_unwanted_symbols()
        x = x.translate(remove_dict)
        return x


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

    def _remove_stopwords(self, df: Optional[np.array]) -> list:
        """Returns all the words in whole dataset after transformation
        """
        try:
            word_list = []
            stop_words = set(stopwords.words('english')) 
            for sent in df:
                for word in sent.lower().split():
                    word = self.preprocess_string(word)
                    if word not in stop_words and word != '':
                        word_list.append(word)
            return word_list
        except Exception as e:
            raise SentimentException(e, sys)

    def split_dataset(self, df: DataFrame):
        try:
            logging.info("Starting Splitting of dataset>>>>")
            X,y = df['text'].values, df['target'].values
            x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y, test_size=self.data_transformation_config.test_size)
            logging.info(f'shape of train data is {x_train.shape}')
            logging.info(f'shape of train data is {x_test.shape}')
            logging.info(">>>>>>> Splitting of dataset Completed")
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise SentimentException(e, sys)
  
    def tockenize(self, x_train, y_train, x_test, y_test, word_list: list):
        """Tokenize and Vectorize using One hot encoding
        """
        try:
            logging.info("Starting Tokenizationm>>>>>>")
            corpus = Counter(word_list)
            # sorting on the basis of most common words
            corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
            # creating a dict
            onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
            
            # tockenize
            final_list_train,final_list_test = [],[]
            for sent in x_train:
                    final_list_train.append([onehot_dict[self.preprocess_string(word)] for word in sent.lower().split()
                                            if self.preprocess_string(word) in onehot_dict.keys()])
            for sent in x_test:
                    final_list_test.append([onehot_dict[self.preprocess_string(word)] for word in sent.lower().split()
                                            if self.preprocess_string(word) in onehot_dict.keys()])
                    
            return np.array(final_list_train), np.array(y_train), np.array(final_list_test), np.array(y_test), onehot_dict
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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            selected_df = self.change_target_values(self.data_validation_artifacts.valid_file_path)
            x_train,x_test,y_train,y_test = self.split_dataset(selected_df[:10000]) # 1600000
            all_words = self._remove_stopwords(x_train)
            x_train,y_train,x_test,y_test,vocab = self.tockenize(x_train,y_train, x_test,y_test, all_words)
            x_train_pad = self.padding_(x_train,self.data_transformation_config.seq_len)
            x_test_pad = self.padding_(x_test,self.data_transformation_config.seq_len)
            logging.info("Saving Transformed data to artifacts>>>")

            os.makedirs(self.data_transformation_config.train_data_dir, exist_ok= True)
            os.makedirs(self.data_transformation_config.test_data_dir, exist_ok= True)

            save_numpy_array_data(self.data_transformation_config.train_data_file_path, x_train_pad)
            save_numpy_array_data(self.data_transformation_config.train_labels_file_path, y_train)
            save_numpy_array_data(self.data_transformation_config.test_data_file_path, x_test_pad)
            save_numpy_array_data(self.data_transformation_config.test_labels_file_path, y_test)
            data_transformation_artifact = DataTransformationArtifact(
                            train_data_file_path = self.data_transformation_config.train_data_file_path,
                            train_labels_file_path = self.data_transformation_config.train_labels_file_path,
                            test_data_file_path = self.data_transformation_config.test_data_file_path,
                            test_labels_file_path =  self.data_transformation_config.test_labels_file_path,
                            vocab = vocab)
            transformation_artifacts_dir = self.data_transformation_config.data_transformation_artifacts_dir
            
            # Save vocab for prediction
            with open(VOCAB_FILE_PATH, 'w') as fp:
                json.dump(vocab, fp)
        
            s3_sync = S3Sync()
            logging.info("Sync transformation files to S3 transformation artifacts folder...") 
            s3_sync.sync_folder_to_s3(folder=transformation_artifacts_dir, aws_bucket_url=S3_ARTIFACTS_URI)
            logging.info("Finished Syncing files to S3 transformation artifacts folder")
            return data_transformation_artifact
        except Exception as e:
            raise SentimentException(e, sys)