from datetime import datetime

import torch

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants
ARTIFACT_DIR = "artifacts"

BUCKET_NAME = "lungxray"

S3_DATA_FOLDER = "data"

CLASS_LABEL_1 = "NORMAL"

CLASS_LABEL_2 = "PNEUMONIA"

DATA_INGESTION_TRAIN_DIR = "train"

DATA_INGESTION_TEST_DIR = "test"

BRIGHTNESS = 0.10

CONTRAST = 0.1

SATURATION = 0.10

HUE = 0.1

RESIZE = 224

CENTERCROP = 224

RANDOMROTATION = 10

NORMALIZE_LIST_1 = [0.485, 0.456, 0.406]

NORMALIZE_LIST_2 = [0.229, 0.224, 0.225]

BATCH_SIZE = 2

SHUFFLE = False

PIN_MEMORY = True

# Model Training Constants
TRAINED_MODEL_DIR = "trained_model"

TRAINED_MODEL_NAME = "model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEP_SIZE = 6

GAMMA = 0.5

EPOCH = 80

# Prediction Constants
PREDICTION_LABEL = {"0": CLASS_LABEL_1, "1": CLASS_LABEL_2}
