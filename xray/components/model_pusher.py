import logging
import sys

from xray.configuration.s3_operations import S3Operation
from xray.entity.artifacts_entity import ModelEvaluationArtifacts, ModelPusherArtifacts
from xray.entity.config_entity import ModelPusherConfig
from xray.exception import XrayException

logger = logging.getLogger(__name__)


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifacts: ModelEvaluationArtifacts,
        s3: S3Operation,
    ):
        self.model_pusher_config = model_pusher_config

        self.s3 = s3

        self.model_evaluation_artifacts = model_evaluation_artifacts

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :    Model pusher artifact
        """
        logger.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            self.s3.upload_file(
                self.model_pusher_config.BEST_MODEL_PATH,
                self.model_pusher_config.S3_MODEL_KEY_PATH,
                self.model_pusher_config.BUCKET_NAME,
                remove=False,
            )
            logger.info("Uploaded best model to s3 bucket")

            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )

            logger.info("Exited the initiate_model_pusher method of ModelTrainer class")

            return model_pusher_artifact

        except Exception as e:
            raise XrayException(e, sys) from e
