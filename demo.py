# from xray.components.data_ingestion import DataIngestion
# from xray.entity.config_entity import DataIngestionConfig
# from xray.configuration.s3_operations import S3Operation
# from xray.constants import *

# data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(), s3_operations= S3Operation())

# data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

# # s3 = S3Operation()

# # s3.read_data_from_s3(ZIP_FILE_NAME, BUCKET_NAME, data_ingestion.ZIP_FILE_DIR)

# from xray.pipeline.train_pipeline import TrainPipeline
from xray.pipeline.prediction_pipeline import PredictionPipeline


# runner = TrainPipeline()

# runner.run_pipeline()



prediction_pipeline = PredictionPipeline()
prediction_pipeline.run_pipeline(r"artifacts/DataIngestionArtifacts/Train/PNEUMONIA/BACTERIA-37006-0001.jpeg")

# from xray.configuration.s3_operations import S3Operation
# import torch
# from xray.models.model import Net

# s3 = S3Operation()

# best_model_path = s3.read_data_from_s3("model.pt","lungxray","artifacts/model.pt")

# model = Net()

# model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

# model.eval()


# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes=image_bytes)

#     outputs = model.forward(tensor)

#     _, y_hat = outputs.max(1)

#     predicted_idx = str(y_hat.item())

#     return class_idx[predicted_idx]