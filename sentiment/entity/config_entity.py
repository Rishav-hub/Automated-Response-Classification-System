from datetime import datetime
import os
from sentiment.constant.training_pipeline import *
from dataclasses import dataclass


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = SOURCE_DIR_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class PredictionPipelineConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    prediction_artifact_dir = os.path.join(ROOT, PREDICTION_PIPELINE_DIR_NAME)
    model_download_path = os.path.join(prediction_artifact_dir, PREDICTION_MODEL_DIR_NAME)

@dataclass
class DataIngestionConfig:
    data_ingestion_artifacts_dir: str = os.path.join(ROOT, training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACTS_DIR)
    s3_file_name:str = S3_FILE_NAME
    bucket_name:str = BUCKET_NAME
    raw_data_dir: str = os.path.join(data_ingestion_artifacts_dir ,RAW_DATA_DIR)
    local_file_path:str = os.path.join(raw_data_dir, s3_file_name)
@dataclass
class DataValidationConfig:
    data_validation_artifacts_dir:str =os.path.join(ROOT, training_pipeline_config.artifact_dir, DATA_VALIDATION_ARTIFACTS_DIR)
    encoding:str = ENCODING
    validated_data_dir:str = os.path.join(data_validation_artifacts_dir, VALIDATED_DATA_DIR)
    validated_file_name: str = os.path.join(validated_data_dir, VALIDATED_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_artifacts_dir:str =os.path.join(ROOT, training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_ARTIFACTS_DIR)
    test_size: float = TEST_SIZE
    seq_len: int = SEQ_LEN
    symbols_to_delete: str = SYMBOLS_TO_DELETE
    train_data_dir: str = os.path.join(data_transformation_artifacts_dir ,DATA_PREPROCESSING_TRAIN_DIR)
    train_data_file_path: str = os.path.join(train_data_dir, TRAIN_DATA_FILE_NAME)
    train_labels_file_path: str = os.path.join(train_data_dir, TRAIN_LABELS_FILE_NAME)
    test_data_dir: str = os.path.join(data_transformation_artifacts_dir ,DATA_PREPROCESSING_TEST_DIR)
    test_data_file_path: str = os.path.join(test_data_dir, TEST_DATA_FILE_NAME)
    test_labels_file_path: str = os.path.join(test_data_dir, TEST_LABELS_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_artifacts_dir: str = os.path.join(ROOT, training_pipeline_config.artifact_dir, MODEL_TRAINING_ARTIFACTS_DIR)
    trained_model_dir: str = os.path.join(model_trainer_artifacts_dir, TRAINED_MODEL_NAME)
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    no_layers: int = NO_LAYERS
    hidden_dim: int = HIDDEN_DIM
    embedding_dim: int = EMBEDDING_DIM
    clip: int = CLIP
    output_dim: int = OUTPUT_DIM

@dataclass
class ModelEvaluationConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    model_evaluation_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR)
    best_model_dir: str = os.path.join(model_evaluation_artifacts_dir, S3_MODEL_DIR_NAME)
    in_channels: int = IN_CHANNELS
    base_accuracy: float = BASE_ACCURACY

@dataclass
class ModelPusherConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    model_pusher_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR)