from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    train_file_path: str 
    test_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str 
    transformed_test_object: str