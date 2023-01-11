from dataclasses import dataclass
from from_root import from_root
from torch.types import Device
from xray.constant.training_pipeline import *
import os


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME

        self.ZIP_FILE_NAME: str = ZIP_FILE_NAME

        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR
        )

        self.TRAIN_DATA_ARTIFACT_DIR = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAIN_DIR
        )

        self.TEST_DATA_ARTIFACT_DIR = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TEST_DIR
        )

        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)

        self.ZIP_FILE_PATH = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME
        )

        self.UNZIPPED_FILE_PATH = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME
        )

        self.PARAMS_TEST_RATIO: int = PARAMS_TEST_RATIO


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR
        )

        self.TRAIN_TRANSFORM_DATA_ARTIFACT_DIR = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TRAIN_DIR
        )

        self.TEST_TRANSFORM_DATA_ARTIFACT_DIR = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TEST_DIR
        )

        self.BRIGHTNESS: float = BRIGHTNESS

        self.CONTRAST: float = CONTRAST

        self.SATURATION: float = SATURATION

        self.HUE: float = HUE

        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION

        self.NORMALIZE_LIST_1: list = NORMALIZE_LIST_1

        self.NORMALIZE_LIST_2: list = NORMALIZE_LIST_2

        self.BATCH_SIZE: int = BATCH_SIZE

        self.SHUFFLE: bool = SHUFFLE

        self.PIN_MEMORY: bool = PIN_MEMORY


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, TRAINED_MODEL_DIR
        )

        self.TRAINED_MODEL_PATH = os.path.join(
            self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME
        )

        self.PARAMS_EPOCHS: int = PARAMS_EPOCHS

        self.STEP_SIZE: int = STEP_SIZE

        self.GAMMA: int = GAMMA

        self.EPOCH: int = EPOCH

        self.MODEL: str = MODEL

        self.OPTIMIZER = OPTIMIZER

        self.DEVICE = DEVICE


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.DEVICE = DEVICE

        self.TEST_LOSS: int = 0

        self.TEST_ACCURACY: int = 0

        self.TOTAL: int = 0

        self.TOTAL_BATCH: int = 0


# Model Pusher Configurations
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, TRAINED_MODEL_DIR
        )

        self.BEST_MODEL_PATH: str = os.path.join(
            self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME
        )

        self.BUCKET_NAME: str = BUCKET_NAME

        self.S3_MODEL_KEY_PATH: str = os.path.join(TRAINED_MODEL_NAME)
