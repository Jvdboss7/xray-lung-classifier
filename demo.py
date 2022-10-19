from xray.components.data_ingestion import DataIngestion
from xray.entity.config_entity import DataIngestionConfig
from xray.configuration.s3_operations import S3Operation
from xray.constants import *

data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(), s3_operations= S3Operation())

data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

# s3 = S3Operation()

# s3.read_data_from_s3(ZIP_FILE_NAME, BUCKET_NAME, data_ingestion.ZIP_FILE_DIR)