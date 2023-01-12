import io
import sys

import numpy as np
import yaml
from PIL import Image

from xray.constant.training_pipeline import *
from xray.exception import XRayException
from xray.logger import logging
from xray.components.data_transformation import DataTransformation


class MainUtils:
    def read_yaml_file(self, filename: str) -> dict:
        logging.info("Entered the read_yaml_file method of MainUtils class")
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise XRayException(e, sys)

    def write_json_to_yaml_file(self, json_file: dict, yaml_file_path: str) -> yaml:
        logging.info("Entered the write_json_to_yaml_file method of MainUtils class")

        try:
            stream = open(yaml_file_path, "w")

            yaml.dump(json_file, stream)

        except Exception as e:
            raise XRayException(e, sys)

    def save_numpy_array_data(self, file_path: str, array: np.array):
        logging.info("Entered the save_numpy_array_data method of MainUtils class")

        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)

            logging.info("Exited the save_numpy_array_data method of MainUtils class")

            return file_path

        except Exception as e:
            raise XRayException(e, sys)

    def load_numpy_array_data(self, file_path: str) -> np.array:
        logging.info("Entered the load_numpy_array_data method of MainUtils class")

        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)

        except Exception as e:
            raise XRayException(e, sys)

    def transform_image(image_bytes):
        try:
            # my_transforms = transforming_training_data()

            my_transforms = ""

            image = Image.open(io.BytesIO(image_bytes))

            return my_transforms(image).unsqueeze(0)

        except Exception as e:
            raise XRayException(e, sys)
