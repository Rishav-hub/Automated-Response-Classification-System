from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    ingested_file_path: str
    ingested_file_name: str   

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_file_path: str

@dataclass
class DataTransformationArtifact:
    train_data_file_path: str
    train_labels_file_path: str
    test_data_file_path: str
    test_labels_file_path: str
    vocab: dict

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_dir_path: str
    trained_model_accuracy : float
    trained_model_loss : float
    vocab: int

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    trained_model_accuracy: float
    s3_model_accuracy: float
    s3_model_path: str
    trained_model_path: str
    
@dataclass
class ModelPusherArtifact:
    pusher_model_dir_path:str
    pusher_tokenizer_dir_path:str
    pusher_target_transform_dir_path:str
    tokenizer_dir_path:str
    target_transform_dir_path:str
    saved_model_dir_path:str

