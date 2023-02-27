import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as Data
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from ast import literal_eval

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_CLASSES = 4  # set equal to output


class MLPNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLPNN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))
        y_pred = self.linear5(x)
        return y_pred


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the label for the current sample
        label = torch.Tensor(self.dataframe['y'].iloc[idx]).to(DEVICE)

        # Get the input for the current sample
        input = torch.Tensor(self.dataframe['x'].iloc[idx]).to(DEVICE)

        return input, label

    def train_test_split(self, train_size=0.8):
        train_size = int(train_size * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])


def train(model, epochs, train_loader, val_loader, log_interval, l_rate):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    train_losses = []
    test_losses = []
    test_scores = []
    best_so_far = np.inf
    counter = 0
    for epoch in range(epochs):
        epoch_val_loss = 0
        train_loss = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            y_train_pred = model(batch_x)
            loss = criterion(y_train_pred, batch_y)
            train_loss.append(loss.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_loss = np.sum(train_loss)/len(train_loader)
        train_losses.append(epoch_train_loss)
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for step, (val_x, val_y) in enumerate(val_loader):
                val_pred = model(val_x)
                loss_val = criterion(val_pred, val_y)
                epoch_val_loss = (epoch_val_loss + loss_val)/len(val_loader)
                labels = torch.argmax(val_y, dim=1).view(-1, 1)
                predicted = torch.argmax(val_pred.data, 1).view(-1, 1)
                y_true.extend(labels.cpu().detach().numpy().tolist())
                y_pred.extend(predicted.cpu().detach().numpy().tolist())
            if epoch_val_loss < best_so_far:
                best_so_far = epoch_val_loss
                counter = 0
            else:
                counter += 1
            if counter > 20:
                break
            test_losses.append(epoch_val_loss.detach().numpy().tolist())
        print("Iteration: ", epoch, " Loss: ", epoch_train_loss, " Validation loss: ", epoch_val_loss)
        test_score = classification_report(y_true, y_pred, zero_division=0, output_dict=False)
        test_scores.append(classification_report(y_true, y_pred, zero_division=0, output_dict=True))

        model.train()

        if (epoch + 1) % log_interval == 0:
            clear_output(wait=True)

            # plot testing loss
            fig, ax = plt.subplots()
            ax.plot(test_losses, label='Testing Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            # plot average f1 score
            fig, ax = plt.subplots()
            ax.plot([score['macro avg']['f1-score'] for score in test_scores], label='Testing F1 Score Macro Avg')
            ax.plot([score['macro avg']['precision'] for score in test_scores],
                    label='Testing Precision Score Macro Avg')
            ax.plot([score['macro avg']['recall'] for score in test_scores], label='Testing Recall Score Macro Avg')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.legend()

            # plot f1 score for each class
            fig, ax = plt.subplots()
            for i in range(N_CLASSES):
                ax.plot([score[str(i)]['f1-score'] for score in test_scores], label=f'Class {i}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.legend()

            plt.show()

            print(
                f"Epoch {epoch + 1}/{epochs}: Training Loss: {epoch_train_loss:.4f} Test Loss: {epoch_val_loss:.4f} \nTest Score:\n {test_score} ")


# dataframe = pd.read_pickle('./signature_condensed_dataframe.pkl')
dataframe = pd.read_csv(filepath_or_buffer='./dataframe_signature.csv')
dataframe.head()
# dataset = CustomDataset(dataframe)
X = dataframe["x"]
Y = dataframe["y"]
data_x, data_y = np.empty((0, len(X.values))), np.empty((0, len(Y.values)))
for row in range(len(X.values)):
    data_x= np.append(data_x, np.array(literal_eval(X[row])), axis=0)
    data_y = np.append(data_y, np.array(literal_eval(Y[row])), axis=0)
X_train, X_tmp, y_train, y_tmp = train_test_split(data_x, data_y, test_size=0.2, random_state=113)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=113)
np.save('X_converted', data_x, allow_pickle =False)
np.save('Y_converted', data_y, allow_pickle =False)
# train_dataset, test_dataset = dataset.train_test_split()
train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = Data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

print("data_x size: ", data_x.shape)
print("data_y size: ", data_y.shape)
print("train dataset size: ", np.array(train_dataset).shape)
print("test dataset size: ", np.array(val_dataset).shape)

D_in, H, D_out = len(data_x), 15, len(data_y)
batch = 1000

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=X_val.shape[0], shuffle=False)

if D_out != N_CLASSES:
    print("Adjust N_CLASSES parameter according to output dimensions!!")
    exit()
print("D_in: ", D_in, "D_out: ", D_out)
epochs = 500
model = MLPNN(D_in, H, D_out)

# train(model=model, epochs=epochs, train_loader=train_loader, val_loader=test_loader, log_interval=epochs, l_rate=1e-4)
