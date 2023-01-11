import io

import os
import sys

import numpy as np
from PIL import Image
from torchvision import transforms

from xray.cloud_storage.s3_operations import S3Operation
from xray.constant.training_pipeline import *
from xray.exception import XRayException
from xray.ml.model import arch

from xray.logger import logging


class PredictionPipeline:
    def __init__(self):
        imsize = 224

        self.loader = transforms.Compose(
            [transforms.Resize(imsize), transforms.ToTensor()]
        )

        self.s3 = S3Operation()

        self.bucket_name = BUCKET_NAME

    def get_train_aug(self) -> torch.Tensor:
        logging.info("Entered the get_train_aug method of PredictionPipeline class")

        try:
            mean = [0.485, 0.456, 0.406]

            std = [0.229, 0.224, 0.225]

            train_aug = transforms.Compose(
                [
                    transforms.Resize(RESIZE),
                    transforms.CenterCrop(CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            logging.info("Exited the get_train_aug method of PredictionPipeline class")

            return train_aug

        except Exception as e:
            raise XRayException(e, sys)

    def image_loader(self, image_bytes) -> Image:
        """load image, returns cuda tensor"""
        logging.info("Entered the image_loader method of PredictionPipeline class")

        try:
            my_transforms = self.get_train_aug()

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0)))

            image = image.reshape(1, 3, 224, 224)

            logging.info("Exited the image_loader method of PredictionPipeline class")

            return image

        except Exception as e:
            raise XRayException(e, sys)

    def get_model_from_s3(self) -> object:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            os.makedirs("artifacts/PredictModel", exist_ok=True)

            predict_model_path = os.path.join(
                os.getcwd(), "artifacts", "PredictModel", TRAINED_MODEL_NAME
            )

            best_model_path = self.s3.read_data_from_s3(
                TRAINED_MODEL_NAME, self.bucket_name, predict_model_path
            )

            logging.info(
                "Exited the get_model_from_s3 method of PredictionPipeline class"
            )

            return best_model_path

        except Exception as e:
            raise XRayException(e, sys)

    def prediction(self, best_model_path: str, image_tensor) -> float:
        logging.info("Entered the prediction method of PredictionPipeline class")

        try:
            model = arch.Net()

            model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

            model.eval()

            outputs = model.forward(image_tensor)

            _, y_hat = outputs.max(1)

            predicted_idx = str(y_hat.item())

            label = PREDICTION_LABEL[predicted_idx]

            logging.info("Exited the prediction method of PredictionPipeline class")

            return label

        except Exception as e:
            raise XRayException(e, sys)

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)

            print(image.shape)

            best_model_path: str = self.get_model_from_s3()

            prediction_label = self.prediction(best_model_path, image)

            logging.info("Exited the run_pipeline method of PredictionPipeline class")

            return prediction_label

        except Exception as e:
            raise XRayException(e, sys)
