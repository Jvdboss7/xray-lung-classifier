import sys

from xray.cloud_storage.s3_operations import S3Operation
from xray.entity.artifacts_entity import ModelPusherArtifact
from xray.entity.config_entity import ModelPusherConfig
from xray.exception import XRayException
from xray.logger import logging


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

        self.s3 = S3Operation()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            self.s3.upload_file(
                self.model_pusher_config.BEST_MODEL_PATH,
                self.model_pusher_config.S3_MODEL_KEY_PATH,
                self.model_pusher_config.BUCKET_NAME,
                remove=False,
            )

            logging.info("Uploaded best model to s3 bucket")

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys) from e
