import sys
import numpy as np
import yaml
from yaml import safe_dump
from xray.constants import *
from xray.exception import XrayException
import logging
from PIL import Image
import io
from xray.components.data_transformation import DataTransformation


# initiatlizing logger
logger = logging.getLogger(__name__)


class MainUtils:
    def read_yaml_file(self, filename: str) -> dict:
        logger.info("Entered the read_yaml_file method of MainUtils class")
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise XrayException(e, sys) from e

    def write_json_to_yaml_file(self, json_file: dict, yaml_file_path: str) -> yaml:
        logger.info("Entered the write_json_to_yaml_file method of MainUtils class")
        try:
            data = json_file
            stream = open(yaml_file_path, "w")
            yaml.dump(data, stream)

        except Exception as e:
            raise XrayException(e, sys) from e

    def save_numpy_array_data(self, file_path: str, array: np.array):
        logger.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logger.info("Exited the save_numpy_array_data method of MainUtils class")
            return file_path

        except Exception as e:
            raise XrayException(e, sys) from e

    def load_numpy_array_data(self, file_path: str) -> np.array:
        logger.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)

        except Exception as e:
            raise XrayException(e, sys) from e

    def transform_image(image_bytes):
        my_transforms = transforming_training_data()

        image = Image.open(io.BytesIO(image_bytes))

        return my_transforms(image).unsqueeze(0)
