from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    train_file_path: str 
    test_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str 
    transformed_test_object: str

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

@dataclass
class ModelEvaluationArtifacts:
    model_accuracy: int 

# Model Pusher Artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str
    s3_model_path: str