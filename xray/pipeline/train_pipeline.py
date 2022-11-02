import torch
from xray.entity.config_entity import (DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    )
from xray.entity.artifacts_entity import (DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts,
    )

from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation
from xray.components.model_training import ModelTrainer
from xray.components.model_evaluation import ModelEvaluation
from xray.components.model_pusher import ModelPusher
from xray.configuration.s3_operations import S3Operation
from xray.models.model import Net
from xray.exception import XrayException
import logging
import sys

logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.s3_operations = S3Operation()

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
                
                data_ingestion_artifact=data_ingestion_artifact,
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


    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        try:
            model_trainer = ModelTrainer(model=Net(),
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise XrayException(e, sys) 

    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifacts, data_ingestion_artifact: DataIngestionArtifacts,
                                    data_transformation_artifact: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
        try:
            use_cuda = torch.cuda.is_available()
            model_evaluation = ModelEvaluation(data_ingestion_artifact= data_ingestion_artifact, data_transformation_artifact = data_transformation_artifact, 
                                                model_evaluation_config=self.model_evaluation_config, model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact

        except Exception as e:
            raise XrayException(e, sys) from e


    def start_model_pusher(self,
        model_evaluation_artifacts: ModelEvaluationArtifacts,s3: S3Operation,) -> ModelPusherArtifacts:
        logger.info("Entered the start_model_pusher method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_evaluation_artifacts=model_evaluation_artifacts,
                s3=s3,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info("Initiated the model pusher")
            logger.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise XrayException(e, sys) from e
        
    def run_pipeline(self) -> None:
        logger.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact, data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_transformation_artifact= data_transformation_artifact
            )
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifacts=model_evaluation_artifact,s3=self.s3_operations,
            )
            logger.info("Exited the run_pipeline method of TrainPipeline class")
            
        except Exception as e:
            raise XrayException(e, sys) from e

