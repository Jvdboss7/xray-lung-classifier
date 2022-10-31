import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xray.entity.config_entity import ModelTrainerConfig
from xray.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from xray.constants import *
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR 
# from xray.models import model
from torchsummary import summary
from tqdm import tqdm
import os

class ModelTrainer:
    def __init__(self,model, data_transformation_artifact: DataTransformationArtifacts,model_trainer_config= ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model = model
        self.device = DEVICE

    def train(self,):
        """
        Description: To train the model 
        
        input: model,device,train_loader,optimizer,epoch 
        
        output: loss, batch id and accuracy
        """
        self.model.train()
        pbar = tqdm(self.data_transformation_artifact.transformed_train_object)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get data
            data, target = data.to(self.device), target.to(self.device)
            # Initialization of gradient
            self.model_trainer_config.OPTIMIZER.zero_grad()
            # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
            # or specific requirements
            ## prediction on data
            y_pred = self.model(data)
            # Calculating loss given the prediction
            loss = F.nll_loss(y_pred, target)
            # Backprop
            loss.backward()
            self.model_trainer_config.OPTIMIZER.step()
            # get the index of the log-probability corresponding to the max value
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')


    def test(self,):
        """
        Description: To test the model
        
        input: model, self.device, test_loader
        
        output: average loss and accuracy
        
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_transformation_artifact.transformed_test_object:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.data_transformation_artifact.transformed_test_object.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(self.data_transformation_artifact.transformed_test_object.dataset),
                100. * correct / len(self.data_transformation_artifact.transformed_test_object.dataset)))

    def initiate_model_trainer(self):
        # Defining the params for training 
        print(self.model)
        model =  self.model_trainer_config.MODEL.to(self.model_trainer_config.DEVICE)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        scheduler = StepLR(optimizer=self.model_trainer_config.OPTIMIZER, step_size=self.model_trainer_config.STEP_SIZE, gamma=self.model_trainer_config.GAMMA)
        # EPOCHS = 4
        # Training the model
        for epoch in range(self.model_trainer_config.EPOCH):
            print("EPOCH:", epoch)
            self.train()
            scheduler.step()
            #print('current Learning Rate: ', self.optimizer.state_dict()["param_groups"][0]["lr"])
            self.test()
        #print(model.state_dict())
        os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(),self.model_trainer_config.TRAINED_MODEL_PATH)
        
        model_trainer_artifact= ModelTrainerArtifacts(trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH)
        return model_trainer_artifact