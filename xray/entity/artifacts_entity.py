from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    train_file_path: str 
    test_file_path: str