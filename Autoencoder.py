#%% 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# %%
dataframe = pd.read_csv('dataframe_signature.csv')
#%%
dataframe.head()
#%% turn dataframe entries into numpy arrays
for column in dataframe.columns:
    dataframe[column] = dataframe[column].apply(lambda  x: eval(x))
#%%
dataframe.head()
# %%
class CustomDataset(Dataset):
    def __init__(self, data:pd.DataFrame):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.data[idx])
        if type(idx) == int:
            return torch.Tensor(self.data.iloc[idx,0]).to(self.device), torch.Tensor(self.data.iloc[idx,1]).to(self.device)
        if type(idx) == list:
            #return pytorch tensor
            return torch.Tensor(self.data.iloc[idx,0]).to(self.device), torch.Tensor(self.data.iloc[idx,1]).to(self.device)
    
    def shuffle(self, SEED = 1):
        self.data = self.data.sample(frac=1, random_state=SEED)
    
    def train_test_split(self, test_size=0.2):
        test_size = int(test_size * len(self.data))
        test_data = self.data[:test_size]
        train_data = self.data[test_size:]
        return train_data, test_data
# %%
dataset = CustomDataset(dataframe)
dataset[[0,1,2]][0]
#%% data loader

train_data, test_data = dataset.train_test_split()
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=3200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3200, shuffle=True)
# %%
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size = 25): 
        super(Autoencoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = nn.Sequential(nn.Linear(input_size, int(hidden_size * 2)), nn.ReLU(),
                                        nn.Linear(int(hidden_size*2), int(hidden_size*1.5)), nn.ReLU(),
                                        nn.Linear(int(hidden_size*1.5), int(hidden_size*1.25)), nn.ReLU(),
                                        nn.Linear(int(hidden_size*1.25), int(hidden_size)), nn.ReLU()) 
        self.decoder = nn.Sequential(nn.Linear(int(hidden_size), int(hidden_size*1.25)), nn.ReLU(),
                                        nn.Linear(int(hidden_size*1.25), int(hidden_size*1.5)), nn.ReLU(),
                                        nn.Linear(int(hidden_size*1.5), int(hidden_size*2)), nn.ReLU(),
                                        nn.Linear(int(hidden_size*2), input_size), nn.Sigmoid())
        self.to(self.device)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def train_model(    
        self,
        train_loader,
        test_loader,
        epochs:int = 100,
        learning_rate:float = 0.001,
        verbose:bool = True):
        self.train()
        train_losses = []
        test_losses = []
        
        train_acc = []
        test_acc = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        for epoch in range(epochs):
            train_loss = 0
            test_loss = 0
            for data in train_loader:
                xs, _ = data
                
                #Drop coordinates stored in first 2 columns
                xs = xs.view(xs.size(0), -1)[:,2:]
                
                # ===================forward=====================
                output = self(xs)
                loss = criterion(output, xs)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ===================log========================
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            train_acc.append(self.test_encoder(train_loader))
            
            for data in test_loader:
                xs, _ = data
                
                #Drop coordinates stored in first 2 columns
                xs = xs.view(xs.size(0), -1)[:,2:]
                
                output = self(xs)
                loss = criterion(output, xs)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            test_acc.append(self.test_encoder(test_loader))
            
            if epoch % 3 == 0 and verbose:
                print('epoch [{}/{}], train_loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, epochs, train_loss, test_loss))
                print('train_acc:{:.4f}, test_acc:{:.4f}'.format(train_acc[-1], test_acc[-1]))
                plt.plot(train_losses, label='train loss')
                plt.plot(test_losses, label='test loss')
                plt.legend()
                plt.show()
                
                plt.plot(train_acc, label='train acc')
                plt.plot(test_acc, label='test acc')
                plt.legend()
                plt.show()
        print('Finished Training')
        return train_losses, test_losses
    def test_encoder(self, loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                xs, _ = data
                
                #Drop coordinates stored in first 2 columns
                xs = xs.view(xs.size(0), -1)[:,2:]
                
                outputs = self(xs)
                #extend xs and outputs to 1d and round outputs
                xs = xs.reshape(-1)
                outputs = outputs.reshape(-1).round()
                total += xs.size(0)
                correct += (xs == outputs).sum().item()
                
        return correct / total
# %%
model = Autoencoder(len(train_dataset[0][0])-2)
# %%
model.train_model(train_loader, test_loader, epochs=100, learning_rate=0.001, verbose=False)
# %% save model state_dict
torch.save(model.state_dict(), "encoder_25.pth")
# %%
