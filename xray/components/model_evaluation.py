import sys
#from pickletools import optimize
import torch 
from torch.nn import CrossEntropyLoss
from xray.components.data_transformation import DataTransformation
from xray.entity.artifacts_entity import ModelEvaluationArtifacts,DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts
from xray.entity.config_entity import ModelEvaluationConfig, DataTransformationConfig

from torch.optim import SGD
from xray.models.model import Net
from xray.exception import XrayException

class ModelEvaluation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifacts, 
                data_transformation_artifact: DataTransformationArtifacts,
                model_evaluation_config: ModelEvaluationConfig,
                model_trainer_artifact: ModelTrainerArtifacts) :

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig(), data_ingestion_artifact=self.data_ingestion_artifact)

    def configuration(self):
        try:
            train_Dataloader, test_DataLoader = self.data_transformation.data_loader()
            print(test_DataLoader)

            model = Net()

            #model.load_state_dict(torch.load('artifacts/TrainedModel/model.pt'))
            load_model_path = self.model_trainer_artifact.trained_model_path
            model.load_state_dict(torch.load(load_model_path))

            model.to(self.model_evaluation_config.DEVICE)


            # net = torch.load('artifacts/training/model.pt').to(self.device)
            cost = CrossEntropyLoss()
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.8)
            model.eval()
            return test_DataLoader, model, cost, optimizer
        except Exception as e:
            raise e

    def test_net(self):
        try:
            test_DataLoader, net, cost, optimizer = self.configuration()

            with torch.no_grad():
                holder = []
                for batch, data in enumerate(test_DataLoader):
                    images = data[0].to(self.model_evaluation_config.DEVICE)
                    labels = data[1].to(self.model_evaluation_config.DEVICE)

                    output = net(images)
                    loss = cost(output, labels)

                    predictions = torch.argmax(output, 1)

                    for i in zip(images, labels, predictions):
                        h = list(i)
                        # h[0] = wandb.Image(h[0])
                        holder.append(h)

                    print(f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}", )

                    self.model_evaluation_config.TEST_LOSS += loss.item()
                    self.model_evaluation_config.TEST_ACCURACY += (predictions == labels).sum().item()
                    self.model_evaluation_config.TOTAL_BATCH += 1
                    self.model_evaluation_config.TOTAL += labels.size(0)

                    print(f"Model  -->   Loss : {self.model_evaluation_config.TEST_LOSS/ self.model_evaluation_config.TOTAL_BATCH} Accuracy : {(self.model_evaluation_config.TEST_ACCURACY / self.model_evaluation_config.TOTAL) * 100} %")
            accuracy = (self.model_evaluation_config.TEST_ACCURACY/ self.model_evaluation_config.TOTAL) * 100
            return accuracy

        except Exception as e:
            raise e

    def initiate_model_evaluation(self):

        try:
            self.configuration()
            accuracy=self.test_net()
            model_evaluation_artifact = ModelEvaluationArtifacts(model_accuracy=accuracy)
            return model_evaluation_artifact
        except Exception as e:
            raise XrayException(e, sys) from e
