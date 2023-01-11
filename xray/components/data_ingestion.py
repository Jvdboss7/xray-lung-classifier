import os
import sys
from zipfile import Path, ZipFile
from xray.entity.config_entity import DataIngestionConfig
from xray.entity.artifacts_entity import DataIngestionArtifacts
from xray.configuration.s3_operations import S3Operation
from xray.exception import XrayException
from xray.constants import *
from PIL import Image
import numpy as np
import shutil
import logging


logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, s3_operations: S3Operation):
        self.data_ingestion_config = data_ingestion_config
        self.s3_operations = s3_operations


    def get_data_from_s3(self) -> None:
        try:
            logger.info("Entered the get_data_from_s3 method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.s3_operations.read_data_from_s3(self.data_ingestion_config.ZIP_FILE_NAME,self.data_ingestion_config.BUCKET_NAME,
                                                self.data_ingestion_config.ZIP_FILE_PATH)
            logger.info("Exited the get_data_from_s3 method of Data ingestion class")
        except Exception as e:
            raise XrayException(e, sys) from e

    def unzip_and_clean(self) -> None:
        logger.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            logger.info("Exited the unzip_and_clean method of Data ingestion class")
        except Exception as e:
            raise XrayException(e, sys) from e

    def train_test_split(self) -> Path:
        """
        This function would split the raw data into train and test folder
        """
        logger.info("Entered the train_test_split method of Data ingestion class")
        try:
            # 1. make train and test folder 
            unzipped_images = self.data_ingestion_config.UNZIPPED_FILE_PATH

            os.makedirs(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, exist_ok=True)
            os.makedirs(self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, exist_ok=True)
            logger.info("Created train data artifacts and test data artifacts directories")
            #params.yaml
            test_ratio = self.data_ingestion_config.PARAMS_TEST_RATIO

            #print(train_path)
            classes_dir = [CLASS_LABEL_1, CLASS_LABEL_2]

            for cls in classes_dir:
                os.makedirs(os.path.join(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, cls), exist_ok= True)
                os.makedirs(os.path.join(self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, cls), exist_ok=True)
                logger.info("Created train data artifacts and test data artifacts directories with class")
            # 2. Split the raw data
            raw_data_path = os.path.join(unzipped_images)

            for cls in classes_dir:
                all_file_names = os.listdir(os.path.join(raw_data_path, cls))
 
                np.random.shuffle(all_file_names)
                train_file_name, test_file_name = np.split(np.array(all_file_names),
                                    [int(len(all_file_names)* (1 - test_ratio))])

                train_names = [os.path.join(raw_data_path, cls, name) for name in train_file_name.tolist()]
                test_names = [os.path.join(raw_data_path, cls, name) for name in test_file_name.tolist()]

                for name in train_names:
                    shutil.copy(name, os.path.join(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, cls))

                for name in test_names:
                    shutil.copy(name, os.path.join(self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, cls))   

            shutil.rmtree(self.data_ingestion_config.UNZIPPED_FILE_PATH, ignore_errors=True)
            logger.info("Exited the train_test_split method of Data ingestion class")

            return self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR

        except Exception as e:
            raise XrayException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts: 
        logger.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try: 
            self.get_data_from_s3()
            self.unzip_and_clean()
            train_file_path, test_file_path = self.train_test_split()
            data_ingestion_artifact = DataIngestionArtifacts(train_file_path=train_file_path, 
                                                                test_file_path=test_file_path)
            logger.info("Exited the initiate_data_ingestion method of Data ingestion class")

            return data_ingestion_artifact

        except Exception as e:
            raise XrayException(e, sys) from e