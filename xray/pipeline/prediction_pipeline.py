from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import io
import sys
import pickle
from xray.exception import XrayException
import os
from xray.configuration.s3_operations import S3Operation
from xray.constants import *
from xray.models.model import Net
from xray.components.data_transformation import DataTransformation
import logging

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        imsize = 224
        self.loader = transforms.Compose([transforms.Resize(imsize),\
                             transforms.ToTensor()])
        self.s3 = S3Operation()
        self.bucket_name = BUCKET_NAME

    def get_train_aug(self) -> torch.Tensor:
        logger.info("Entered the get_train_aug method of PredictionPipeline class")
        try:
            mean = [0.485, 0.456, 0.406]

            std = [0.229, 0.224, 0.225]

            train_aug = transforms.Compose(
                [
                    transforms.Resize(RESIZE),
                    transforms.CenterCrop(CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
            logger.info("Exited the get_train_aug method of PredictionPipeline class")
            return train_aug
            
        except Exception as e:
            raise XrayException(e, sys) from e

    def image_loader(self, image_bytes) -> Image:
        """load image, returns cuda tensor"""
        logger.info("Entered the image_loader method of PredictionPipeline class")
        try:
            my_transforms = self.get_train_aug()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0)))
            image = image.reshape(1, 3, 224, 224)
            logger.info("Exited the image_loader method of PredictionPipeline class")
            return image

        except Exception as e:
            raise XrayException(e, sys) from e

    def get_model_from_s3(self) -> object:

        """
        Method Name :   predict
        Description :   This method predicts the image. 
        
        Output      :   Predictions 
        """
        logger.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket     
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(),"artifacts", "PredictModel", TRAINED_MODEL_NAME)
            best_model_path = self.s3.read_data_from_s3(TRAINED_MODEL_NAME,self.bucket_name,predict_model_path)
            logger.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            return best_model_path
        
        except Exception as e:
            raise XrayException(e, sys) from e
    
    def prediction(self, best_model_path: str, image_tensor) -> float:
        logger.info("Entered the prediction method of PredictionPipeline class")
        try:
            model = Net()
            model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
            model.eval()
            outputs = model.forward(image_tensor)
            _, y_hat = outputs.max(1)
            predicted_idx = str(y_hat.item())
            label = PREDICTION_LABEL[predicted_idx]
            logger.info("Exited the prediction method of PredictionPipeline class")
            return label

        except Exception as e:
            raise XrayException(e, sys) from e       

    def run_pipeline(self, data):
        logger.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)
            print(image.shape)
            best_model_path: str = self.get_model_from_s3()
            prediction_label = self.prediction(best_model_path, image)
            logger.info("Exited the run_pipeline method of PredictionPipeline class")
            return prediction_label
        except Exception as e:
            raise XrayException(e, sys) from e