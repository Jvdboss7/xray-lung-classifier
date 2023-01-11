import os
import sys

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xray.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
)
from xray.entity.config_entity import DataTransformationConfig
from xray.exception import XRayException

from xray.logger import logging


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifacts,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def transforming_training_data(self):
        logging.info(
            "Entered the transforming_training_data method of Data transformation class"
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize(self.data_transformation_config.RESIZE),
                transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                transforms.ColorJitter(
                    brightness=self.data_transformation_config.BRIGHTNESS,
                    contrast=self.data_transformation_config.CONTRAST,
                    saturation=self.data_transformation_config.SATURATION,
                    hue=self.data_transformation_config.HUE,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(
                    self.data_transformation_config.RANDOMROTATION
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.data_transformation_config.NORMALIZE_LIST_1,
                    self.data_transformation_config.NORMALIZE_LIST_2,
                ),
            ]
        )

        logging.info(
            "Exited the transforming_training_data method of Data transformation class"
        )

        return train_transform

    def transforming_testing_data(self):
        logging.info(
            "Entered the transforming_testing_data method of Data transformation class"
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(self.data_transformation_config.RESIZE),
                transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.data_transformation_config.NORMALIZE_LIST_1,
                    self.data_transformation_config.NORMALIZE_LIST_2,
                ),
            ]
        )

        logging.info(
            "Entered the transforming_testing_data method of Data transformation class"
        )

        return test_transform

    def data_loader(self):
        logging.info("Entered the data_loader method of Data transformation class")

        train_transform = self.transforming_training_data()

        test_transform = self.transforming_testing_data()

        train_data = ImageFolder(
            os.path.join(self.data_ingestion_artifact.train_file_path),
            transform=train_transform,
        )

        test_data = ImageFolder(
            os.path.join(self.data_ingestion_artifact.test_file_path),
            transform=test_transform,
        )

        logging.info("Created train data and test data paths")

        train_loader = DataLoader(
            train_data,
            batch_size=self.data_transformation_config.BATCH_SIZE,
            shuffle=self.data_transformation_config.SHUFFLE,
            pin_memory=self.data_transformation_config.PIN_MEMORY,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=self.data_transformation_config.BATCH_SIZE,
            shuffle=self.data_transformation_config.SHUFFLE,
            pin_memory=self.data_transformation_config.PIN_MEMORY,
        )

        class_names = train_data.classes

        print(class_names)

        print(f"Number of train images: {len(train_data)}")

        print(f"Number of test images: {len(test_data)}")

        print(train_loader)

        logging.info("Exited the data_loader method of Data transformation class")

        return train_loader, test_loader

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info(
                "Entered the initiate_data_transformation method of Data transformation class"
            )

            self.transforming_training_data()

            self.transforming_testing_data()

            train_loader, test_loader = self.data_loader()

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
            )

            logging.info(
                "Exited the initiate_data_transformation method of Data transformation class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
