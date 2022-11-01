from xray.models.model import Net 
import torch
from torchsummary import summary


# Data Ingestion Constants
ARTIFACTS_DIR = 'artifacts'
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'xray.log'

BUCKET_NAME = 'lungxray'
ZIP_FILE_NAME = 'chest_xray.zip'
CLASS_LABEL_1 = 'NORMAL'
CLASS_LABEL_2 = 'PNEUMONIA'
RAW_FILE_NAME = 'chest_xray'

DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'Train'
DATA_INGESTION_TEST_DIR = 'Test'
PARAMS_TEST_RATIO = 0.2

# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_DIR = 'Train'
DATA_TRANSFORMATION_TEST_DIR = 'Test'

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
PIN_MEMORY= True

# Model Training Constants 
TRAINED_MODEL_DIR = 'TrainedModel'
TRAINED_MODEL_NAME = 'model.pt'
MODEL = Net()
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.8)

PARAMS_EPOCHS = 3
STEP_SIZE = 6
GAMMA = 0.5
EPOCH = 1
