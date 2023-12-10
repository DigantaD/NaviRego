import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import torch
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

wandb_cred = '4eb778ee5ccad70bef5ee7f1f9af7537f90b107e'
wandb.login(key=wandb_cred)
wandb.init(project='navirego', name='training-run')

class TopicGraphModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(TopicGraphModel, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Apply the second GCN layer (output layer)
        x = self.conv2(x, edge_index)

        return x

class TrainerNetwork():

    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder

    def data_transactions(self):
        X = self.data.x.numpy()
        Y = self.data.y.numpy()
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        train_data = Data(x=X_train, y=Y_train, edge_index=data.edge_index)
        val_data = Data(x=X_val, y=Y_val, edge_index=data.edge_index)
        test_data = Data(x=X_test, y=Y_test, edge_index=data.edge_index)

        return train_data, val_data, test_data

    def create_model(self):
        in_channels = 1000
        hidden_channels = 256
        num_classes = len(self.label_encoder.classes_)
        self.model = TopicGraphModel(in_channels, hidden_channels, num_classes)

    def train(self, train_data, val_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
        # Path to save the best model
        best_model_path = os.path.join(self.save_root, 'NaviRego-Best-V1.pth')
        num_epochs = 10000
        best_val_loss = float('inf')

        for epoch in tqdm(range(1, num_epochs + 1)):
    
            self.model.train()
            
            inputs = train_data.to(device)
            optimizer.zero_grad()
            output = self.model(inputs)
            loss = criterion(output, train_data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Decoding logits to obtain predicted class labels
            probabilities = nn.functional.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            
            correct_predictions = (predicted_labels == train_data.y).sum().item()
            accuracy = correct_predictions / len(train_data.y)
            
            self.model.eval()
            
            with torch.no_grad():
                inputs = val_data.to(device)
                output = self.model(inputs)
                val_loss = criterion(output, val_data.y)

                # Decoding logits to obtain predicted class labels
                probabilities = nn.functional.softmax(output, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)

                correct_predictions = (predicted_labels == val_data.y).sum().item()
                val_accuracy = correct_predictions / len(val_data.y)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.to('cpu')
                torch.save(self.model.state_dict(), best_model_path)

            scheduler.step()

            wandb.log({
                'epoch': epoch, 
                'loss': loss.item(), 
                'accuracy': accuracy,
                'val_loss': val_loss.item(),
                'val_accuracy': val_accuracy
            })

        return best_model_path

    def evaluate(self, test_data):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            inputs = test_data.to(device)
            output = self.model(inputs)
            test_loss = criterion(output, val_data.y)

            # Decoding logits to obtain predicted class labels
            probabilities = nn.functional.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            correct_predictions = (predicted_labels == val_data.y).sum().item()
            test_accuracy = correct_predictions / len(val_data.y)

        wandb.log({
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy
        })

        self.model.to('cpu')

    def process(self):
        train_data, val_data, test_data = self.data_transactions()
        self.create_model()
        best_model_path = self.train(train_data, val_data)
        self.evaluate(test_data)
        return best_model_path