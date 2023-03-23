import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import torch.optim as optim
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
from IPython.display import clear_output


# NN model for earthquake damage evaluation
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPICENTER = [13.466489, 42.366599]
TRAIN = False
ORIGINAL_PROPERTIES = {'Latitude':{'mean': 42.316818950803125,
                                    'std': 0.13075990982450972,
                                    'max': 42.7362168,
                                    'min': 41.7158661414168
                                    },
                        'Longitude':{'mean': 13.521897196462286,
                                    'std': 0.1894015325303454,
                                    'max': 14.2035408,
                                    'min': 13.0
                                    }
                        }


class FullyConnectedNN(nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, depth):
        super(FullyConnectedNN, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.fc_layers = nn.Sequential()

        for i in range(depth):
            in_features = self.input_len if i == 0 else self.hidden_dim
            self.fc_layers.add_module(f"fc{i}", nn.Linear(in_features, self.hidden_dim))
            self.fc_layers.add_module(f"relu{i}", nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim, self.output_len), nn.Softmax(dim=1))
        self.to(DEVICE)
        # print number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x):
        x = self.fc_layers(x.view(-1, self.input_len))
        x = self.classifier(x)
        return x


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



class BalancedBCELoss(nn.Module):
    def __init__(self, alpha, reduction='elementwise_mean'):
        super(BalancedBCELoss, self).__init__()
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.ones(N_CLASSES, dtype=torch.float32).to(DEVICE)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32).to(DEVICE)

    def forward(self, inputs, targets, alpha=None):
        if alpha != None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32).to(DEVICE)

        BCE_loss = torch.clamp(targets * inputs, 0.000001, 1)

        F_loss = -(self.alpha.repeat(targets.shape[0], 1) * BCE_loss.log())
        if self.reduction == 'elementwise_mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def plot_risk_map(model, test_loader):
    model.eval()
    grid = np.zeros((60, 60))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            for x in range(-30, 30, 1):
                for y in range(-30, 30, 1):
                    inputs[:, 0] = y / 10  # LATITUDE
                    inputs[:, 1] = x / 10  # LONGITUDE
                    outputs = model(inputs)
                    classes = torch.argmax(outputs, 1)
                    # map classes to colors
                    damages = class_to_color(classes)
                    site_damage = torch.mean(damages)
                    grid[
                        y + 30, x + 30] = site_damage  # <----------- be careful with the order of x and y (CHECK THIS)
                    # print(outputs)
            return grid


def plot_risk_map_unnormalized(model, loader, grid_size=0.01):
    model.eval()
    x_min = ORIGINAL_PROPERTIES['Longitude']['min']
    x_max = ORIGINAL_PROPERTIES['Longitude']['max']
    y_min = ORIGINAL_PROPERTIES['Latitude']['min']
    y_max = ORIGINAL_PROPERTIES['Latitude']['max']
    grid_size = grid_size

    # create a 2D grid
    xs = np.arange(x_min, x_max + grid_size, grid_size)
    ys = np.arange(y_min, y_max + grid_size, grid_size)
    xyzs = []
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            for x in xs:
                for y in ys:
                    inputs[:, 0] = (y - ORIGINAL_PROPERTIES['Latitude']['mean']) / ORIGINAL_PROPERTIES['Latitude'][
                        'std']
                    inputs[:, 1] = (x - ORIGINAL_PROPERTIES['Longitude']['mean']) / ORIGINAL_PROPERTIES['Longitude'][
                        'std']
                    outputs = model(inputs)
                    classes = torch.argmax(outputs, 1)
                    # map classes to colors
                    damages = class_to_color(classes)
                    site_damage = torch.mean(damages)
                    xyzs.append([x, y, site_damage])
            break

    xyzs = np.array(xyzs)
    # 3d plot using xyzs
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_title('Risk Map')
    # change perspective
    # ax.view_init(55, 200)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Risk Score')
    # remove z ticks
    ax.set_zticks([])
    ax.scatter3D(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], c=xyzs[:, 2], cmap='viridis', s=5)
    # plot a marker for the epicenter
    # ax.scatter3D(EPICENTER[1], EPICENTER[0], np.arange( 0, 3, 0.05), c='red', marker='.', s=100)
    plt.show()

    # 2d variant
    for data in loader:
        inputs, labels = data
        sites_ys = inputs[:, 0] * ORIGINAL_PROPERTIES['Latitude']['std'] + ORIGINAL_PROPERTIES['Latitude']['mean']
        sites_xs = inputs[:, 1] * ORIGINAL_PROPERTIES['Longitude']['std'] + ORIGINAL_PROPERTIES['Longitude']['mean']
        labels = torch.argmax(labels, 1).detach().cpu()
        colors = class_to_color(labels)

        plt.figure(figsize=(20, 20))
        plt.title('Risk Map')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.scatter(xyzs[:, 0], xyzs[:, 1], c=xyzs[:, 2], cmap='viridis', s=200)
        plt.scatter(EPICENTER[0], EPICENTER[1], c='red', marker='x', s=300, label='Epicenter')
        plt.scatter(sites_xs.detach().cpu(), sites_ys.detach().cpu(), c=colors, marker='.', s=25, label='Sites',
                    cmap='PuRd', alpha=0.75)
        # flip y and x axis
        plt.colorbar()
        plt.legend()
        plt.show()
        break

    return np.array(xyzs)


def class_to_color(classes):
    # take the classes and map them to colors from the unique_values_dict
    colors = torch.zeros(classes.shape)
    for i, class_ in enumerate(classes):
        colors[i] = unique_values_dict['sez4_danno_strutturale_copertura'][class_]
    return colors.float().detach().cpu()


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, num_epochs, log_interval, PLOT_MAP, N_CLASSES):
    model.to(DEVICE)
    train_losses = []
    test_losses = []
    test_scores = []

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            y_true = []
            y_pred = []
            for data in test_dataloader:
                inputs, labels = data

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

                labels = torch.argmax(labels, dim=1).view(-1, 1)
                predicted = torch.argmax(outputs.data, 1).view(-1, 1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(predicted.cpu().numpy().tolist())

        test_loss /= len(test_dataloader)
        test_losses.append(test_loss)
        test_score = classification_report(y_true, y_pred, zero_division=0, output_dict=False)

        test_scores.append(classification_report(y_true, y_pred, zero_division=0, output_dict=True))

        if (epoch + 1) % log_interval == 0:
            clear_output(wait=True)

            # plot training loss
            fig, ax = plt.subplots()
            ax.plot(train_losses, label='Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

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

            # # plot average precision score
            # fig, ax = plt.subplots()
            # ax.plot([score['macro avg']['precision'] for score in test_scores], label='Testing Precision Score Macro Avg')
            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('Precision Score')
            # ax.legend()

            # # plot average recall score
            # fig, ax = plt.subplots()
            # ax.plot([score['macro avg']['recall'] for score in test_scores], label='Testing Recall Score Macro Avg')
            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('Recall Score')
            # ax.legend()

            # plot f1 score for each class
            fig, ax = plt.subplots()
            for i in range(N_CLASSES):
                ax.plot([score[str(i)]['f1-score'] for score in test_scores], label=f'Class {i}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.legend()
            plt.show()

            if PLOT_MAP:
                plot_risk_map_unnormalized(model, test_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}: Training Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} \nTest Score:\n {test_score}")


def study_dummy_columns(model, loader, unique_values_dict, PLOT=True, VERBOSE=False):
    model.eval()
    shapes = [len(v) for v in unique_values_dict.values()]

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            categorical_inputs = inputs[:, 2:]  # remove lat and long

            scores = {key: {} for key in unique_values_dict.keys()}

            for i, column in enumerate(unique_values_dict.keys()):  # for each column

                if i != len(unique_values_dict.keys()) - 1:
                    if VERBOSE:
                        print('\n Column Name: ', column)
                    offset = sum(shapes[:i])
                    column_onehot = categorical_inputs[:, offset:offset + shapes[i]]
                    dummy_categorical_signatures = categorical_inputs.clone()

                    for j in range(column_onehot.shape[1]):  # for each unique value in the column
                        if VERBOSE:
                            print('Value Name:', unique_values_dict[column][j])
                        dummy_onehot = torch.zeros_like(column_onehot)

                        dummy_onehot[:, j] = 1
                        dummy_categorical_signatures[:, offset:offset + shapes[i]] = dummy_onehot

                        dummy_input = torch.cat((inputs[:, :2], dummy_categorical_signatures), dim=1)

                        # calculate dummy output
                        outputs = model(dummy_input)
                        classes = torch.argmax(outputs, 1)

                        # map classes to colors
                        damages = class_to_color(classes)
                        site_damage = torch.mean(damages)

                        scores[column][unique_values_dict[column][j]] = site_damage.item()

            if PLOT:
                for column in scores.keys():
                    fig, ax = plt.subplots()
                    fig.set_figheight(5)
                    fig.set_figwidth(20)

                    # sort scores by value
                    scores[column] = {k: v for k, v in sorted(scores[column].items(), key=lambda item: item[1])}

                    # sort scores by key alphabetically
                    # scores[column] = {k: v for k, v in sorted(scores[column].items(), key=lambda item: str(item[0]))}

                    # plot scores
                    for key in scores[column].keys():
                        ax.bar(str(key), scores[column][key])
                    ax.set_xlabel(column)
                    ax.set_ylabel('Mean Damage')
                    ax.set_title(f'Mean Damage for each value in {column}')
                    plt.show()

            return scores



def study_dummy_columns_gtruth(model, loader, unique_values_dict, PLOT=True, VERBOSE=False):
    model.eval()
    shapes = [len(v) for v in unique_values_dict.values()]

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            categorical_inputs = inputs[:, 2:]  # remove lat and long

            scores = {key: {} for key in unique_values_dict.keys()}

            for i, column in enumerate(unique_values_dict.keys()):  # for each column

                if i != len(unique_values_dict.keys()) - 1:
                    if VERBOSE:
                        print('\n Column Name: ', column)
                    offset = sum(shapes[:i])
                    column_onehot = categorical_inputs[:, offset:offset + shapes[i]]
                    dummy_categorical_signatures = categorical_inputs.clone()

                    for j in range(column_onehot.shape[1]):  # for each unique value in the column
                        if VERBOSE:
                            print('Value Name:', unique_values_dict[column][j])
                        dummy_onehot = torch.zeros_like(column_onehot)

                        dummy_onehot[:, j] = 1
                        dummy_categorical_signatures[:, offset:offset + shapes[i]] = dummy_onehot

                        dummy_input = torch.cat((inputs[:, :2], dummy_categorical_signatures), dim=1)

                        # calculate dummy output
                        outputs = model(dummy_input)
                        classes = torch.argmax(outputs, 1)

                        # map classes to colors
                        damages = class_to_color(classes)
                        site_damage = torch.mean(damages)

                        scores[column][unique_values_dict[column][j]] = site_damage.item()

            if PLOT:
                for column in scores.keys():
                    fig, ax = plt.subplots()
                    fig.set_figheight(5)
                    fig.set_figwidth(20)

                    # sort scores by value
                    scores[column] = {k: v for k, v in sorted(scores[column].items(), key=lambda item: item[1])}

                    # sort scores by key alphabetically
                    # scores[column] = {k: v for k, v in sorted(scores[column].items(), key=lambda item: str(item[0]))}

                    # plot scores
                    for key in scores[column].keys():
                        ax.bar(str(key), scores[column][key])
                    ax.set_xlabel(column)
                    ax.set_ylabel('Mean Damage')
                    ax.set_title(f'Mean Damage for each value in {column}')
                    plt.show()

            return scores


# dataframe = pd.read_csv('dataframe_signature.csv')
dataframe = pd.read_pickle('./signature_scalar_condensed_dataframe_copertura_4cat.pkl')

dataset = CustomDataset(dataframe)
train_dataset, test_dataset = dataset.train_test_split()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=True)

# Get weights for each class for balancing loss function
N_CLASSES = len(dataset[0][1])

# # get support of each class in dataframe
# scalar_df = dataframe['y'].apply(lambda x: np.argmax(x))
# weights = scalar_df.value_counts()
#
# # sort weights by key
# weights = weights.sort_index()
# weights = weights.to_numpy()
# weights = torch.Tensor(weights).to(DEVICE)
#
# ALPHA = torch.ones(N_CLASSES).to(DEVICE)/weights
#
# # clamp alpha to avoid exessive weight on rare classes
# ALPHA = ALPHA.clamp(min(ALPHA).item(), max(ALPHA).item())  # multiply by 0.x to avoid overfitting
#
# # normalize alpha
# ALPHA = ALPHA/ALPHA.sum()

# turn ALPHA into all ones
ALPHA = torch.ones(N_CLASSES).to(DEVICE)

unique_values_dict = pickle.load(open('unique_values_dict_scalar_scores_copertura_4cat.pkl', 'rb'))

model = FullyConnectedNN(input_len=len(dataset[0][0]), output_len=len(dataset[0][1]), hidden_dim=25, depth=6)

# model = pickle.load(open('model.pkl', 'rb'))
if(TRAIN):
    train(model, train_loader, test_loader, BalancedBCELoss(alpha=ALPHA), optim.Adam(model.parameters(), lr=0.001),
          num_epochs=200, log_interval=200, PLOT_MAP = False , N_CLASSES=N_CLASSES)

    # save model
    pickle.dump(model, open('model_lr5_copertura_4cat.pkl', 'wb'))

if not (TRAIN):
    model = pickle.load(open('model_lr5_copertura_4cat.pkl', 'rb'))

    # plot_risk_map_unnormalized(model, test_loader)

    scores = study_dummy_columns(model, test_loader, unique_values_dict)