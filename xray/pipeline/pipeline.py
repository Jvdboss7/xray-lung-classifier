from xray.entity.config_entity import (DataIngestionConfig,
    DataTransformationConfig
    )
from xray.entity.artifacts_entity import (DataIngestionArtifacts,
    DataTransformationArtifacts
    )

from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation
from xray.configuration.s3_operations import S3Operation
from xray.exception import XrayException
import logging
import sys

logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()


    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logger.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config, s3_operations= S3Operation()
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the train_set and test_set from s3")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise XrayException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifacts) -> DataTransformationArtifacts:
        logger.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )
        try:
            data_transformation = DataTransformation(
                #data_ingestion_artifacts=data_ingestion_artifact,
                data_ingestion_config = self.data_ingestion_config,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logger.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise XrayException(e, sys) from e

        
    def run_pipeline(self) -> None:
        logger.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
        except Exception as e:
            raise XrayException(e, sys) from e