import sys

from sentiment.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,\
                            DataValidationConfig,DataTransformationConfig, ModelTrainerConfig,\
                            ModelEvaluationConfig, ModelPusherConfig
from sentiment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, \
                            DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from sentiment.exception import SentimentException
from sentiment.logger import logging
from sentiment.components.data_ingestion import DataIngestion
from sentiment.components.data_validation import DataValidation
from sentiment.components.data_transformation import DataTransformation
from sentiment.components.model_trainer import ModelTrainer
from sentiment.components.model_evaluation import ModelEvaluation
from sentiment.components.model_pusher import ModelPusher
from sentiment.configuration.s3_operations import S3Operation

class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_operations = S3Operation()
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
       # self.s3_sync = S3Sync()
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion>>>>>>>>")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config, s3_operations= S3Operation())
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            print("Completed Data Ingestion")
            logging.info("<<<<<<<<Starting data ingestion")
            return data_ingestion_artifact
        except Exception as e:
            raise SentimentException(e,sys)

    def start_data_validaton(self,data_ingestion_artifacts:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("Started Data Validation>>>>>>>>>")
            data_validation = DataValidation(data_validation_config = self.data_validation_config,\
                                            data_ingestion_artifacts=data_ingestion_artifacts)
            data_validation_artifact = data_validation.initiate_data_validation()
            print("Completed Data Validation")
            logging.info("<<<<<<<<Completed Data Validation")
            return data_validation_artifact
        except Exception as e:
            raise SentimentException(e,sys)

    def start_data_transformation(self,data_validation_artifacts:DataValidationArtifact):
        try:
            data_transformation = DataTransformation(data_validation_artifacts=data_validation_artifacts,\
                            data_transformation_config=self.data_transformation_config)
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except  Exception as e:
            raise  SentimentException(e,sys)
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer = ModelTrainer(model_trainer_config = self.model_trainer_config,\
                             data_transformation_artifact = data_transformation_artifact,)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except  Exception as e:
            raise  SentimentException(e,sys)

    def start_model_evaluation(self,model_trainer_artifact:ModelTrainerArtifact):
        try:
            model_eval = ModelEvaluation(self.model_evaluation_config, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise SentimentException(e,sys)

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise  SentimentException(e,sys)

    def run_pipeline(self):
        try:
            
            TrainPipeline.is_pipeline_running=True

            data_ingestion_artifact:DataIngestionArtifact= self.start_data_ingestion()
            data_validation_artifacts:DataValidationArtifact= self.start_data_validaton(data_ingestion_artifacts= data_ingestion_artifact)
            if not data_validation_artifacts.validation_status:
                logging.info("Ending Training pipeline as Data Validation Failed")
                raise Exception("Data Format is not correct >>>>")
            data_transformation_artifact = self.start_data_transformation(data_validation_artifacts=data_validation_artifacts)

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(model_trainer_artifact)
            if not model_eval_artifact.is_model_accepted:
                return {"status": False, "msg": "Trained model is not better than the best model"}
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            print(model_pusher_artifact)
            TrainPipeline.is_pipeline_running=False
            return {"status": True, "msg": "Training Completed!!!"}
        except  Exception as e:
            #self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise SentimentException(e,sys)

