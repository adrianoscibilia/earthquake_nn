#%%
# import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from nltk import flatten
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle


dataframe = pd.read_excel('aquila.xls')
dataframe.head()

len(dataframe.columns)

#%% turn coordinates to float and normalize ################################################
coordinates_columns = ['coordinate_lat', 'coordinate_lon']


def coord_to_float(coord):
    coord = str(coord)
    if coord != 'nan':
        # split after dot and take first element
        coord = coord.split('.')[0]
        # add a dot after second number
        coord = coord[:2] + '.' + coord[2:]
        return float(coord)
    else:
        return float(coord)


PROPERTIES = []
for column in coordinates_columns:
    dataframe[column] = dataframe[column].apply(coord_to_float)
    # drop rows with 0.0 coordinates
    dataframe = dataframe.dropna(subset=[column])

    # normalize coordinates
    mean_coord = dataframe[column].mean()
    std_coord = dataframe[column].std()
    data_properties = {'mean': mean_coord, 'std': std_coord, 'max': dataframe[column].max(),
                       'min': dataframe[column].min()}
    PROPERTIES.append(data_properties)
    dataframe[column] = (dataframe[column] - mean_coord) / std_coord

    print(dataframe[column].head())

# %% Select columns ##########################################################
x_columns = ['coordinate_lat',
    'coordinate_lon',
    'identificativoposizioneedificio',
    # 'sez3_regolarita2',
    # 'sez3_rinforzata',
    'sez3_struttura_orizzontale_1',
    'sez2_altezzamediapiano',
    'sez2_pianiinterrati',
    # 'sez3_mista',
    'sez3_struttura_verticale_1',
    'sez2_numeropiani',
    'sez3_catene_o_cordoli_1',
    'sez2_superficiepiano',
    # 'sez3_regolarita1',
    'sez3_pilastriisolati',
    # 'sez3_struttura_verticale_2',
    'sez2_costruzioneristrutturazione1',
    # 'sez3_catene_o_cordoli_2',
    # 'sez3_struttura_orizzontale_2',
    # 'sez2_costruzioneristrutturazione2',
    'sez7_morfologia_versante'
    ]
# y_columns = ['sez4_danno_strutturale_strutture_verticali',
#     'sez4_danno_strutturale_scale',
#     'sez4_danno_strutturale_tamponature_tramezzi',
#     'sez4_danno_strutturale_copertura',
#     'sez4_danno_strutturale_solai'
#     ]
chosen_damage = 'sez4_danno_strutturale_strutture_verticali'
y_columns = [chosen_damage]

Y_SIZE = len(y_columns)
# reorganize dataframe ##########################################################
dataframe_x = dataframe[x_columns]
dataframe_y = dataframe[y_columns]
dataframe = pd.concat([dataframe_x, dataframe_y], axis=1)
# print(len(dataframe.columns))
# dataframe.head(20)

# replace nan values within columns of interest
dataframe['sez3_struttura_orizzontale_1'] = dataframe['sez3_struttura_orizzontale_1'].fillna('Non identificata')
dataframe['sez3_catene_o_cordoli_1'] = dataframe['sez3_catene_o_cordoli_1'].fillna('no')
dataframe[chosen_damage] = dataframe[chosen_damage].fillna('Danno Nullo')

# %% # count nan in columns
dataframe = dataframe.dropna()
for column in dataframe.columns:
    print(column, dataframe[column].isnull().sum())

condensed_categories = {
    'Danno Nullo': 'Danno Nullo',

    'Danno D1 Leggero:<1/3': 'Danno Leggero',
    'Danno D1 Leggero:1/3-2/3': 'Danno Leggero',
    'Danno D1 Leggero:>2/3': 'Danno Leggero',

    'Danno D2-D3 Medio-Grave:<1/3': 'Danno Medio',
    'Danno D2-D3 Medio-Grave:1/3-2/3': 'Danno Medio',
    'Danno D2-D3 Medio-Grave:>2/3': 'Danno Medio',
    'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3': 'Danno Medio',
    'Danno D2-D3 Medio-Grave:1/3-2/3, Danno D1 Leggero:<1/3': 'Danno Medio',
    'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:>2/3': 'Danno Medio',
    'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:1/3-2/3': 'Danno Medio',

    'Danno D4-D5 Gravissimo:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:1/3-2/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:>2/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:1/3-2/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:>2/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:1/3-2/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:1/3-2/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:>2/3, Danno D2-D3 Medio-Grave:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:>2/3, Danno D1 Leggero:<1/3': 'Danno Grave',
    'Danno D4-D5 Gravissimo:1/3-2/3, Danno D1 Leggero:1/3-2/3': 'Danno Grave'
}

final_categories = {
    'Danno Nullo': 0,
    'Danno Leggero': 1,
    'Danno Medio': 2,
    'Danno Grave': 3
}

# map categories ###############################################################
dataframe[chosen_damage] = dataframe[chosen_damage].map(condensed_categories)
dataframe[chosen_damage] = dataframe[chosen_damage].map(final_categories)
dataframe[chosen_damage].value_counts()

unique_values_dict = {}
for column in dataframe.columns:
    if column in coordinates_columns:
        continue
    column_uniques = dataframe[column].unique() if type(dataframe[column]) == pd.Series else dataframe[column].iloc[:,
                                                                                             0].unique()
    unique_values_dict[column] = column_uniques

#  sort unique values
unique_values_dict[chosen_damage] = sorted(
    unique_values_dict[chosen_damage])

# %% display unique values for each column and save as pkl
# print(unique_values_dict)
with open('unique_values_dict_scalar_scores_strutture_verticali_4cat.pkl', 'wb') as f:
    pickle.dump(unique_values_dict, f)

dataframe[y_columns[0]].unique()

#Count types of damage
dataframe[y_columns[0]].value_counts()

# %% Transform data into onehot encoding ##############################################################
def find_unique_positions(column, x, unique_values_dict):
    return  list(unique_values_dict[column]).index(x)
def int_to_onehot(column, x, unique_values_dict):
    onehot = [0]*(len(unique_values_dict[column]))
    onehot[x] = 1
    return onehot

def turn_to_numeric(dataframe, columns_exeptions = [], unique_values_dict = {}):
    for column in dataframe.columns:
        if column in columns_exeptions:
            continue
        print(column)
        dataframe[column] = dataframe[column].apply(lambda x: find_unique_positions(column=column, x=x, unique_values_dict=unique_values_dict))
    return dataframe
def numeric_to_onehot(dataframe, columns_exeptions = [], unique_values_dict = {}):
    for column in dataframe.columns:
        if column in columns_exeptions:
            continue
        dataframe[column] = dataframe[column].apply(lambda x: int_to_onehot(column=column, x=x, unique_values_dict=unique_values_dict))
    return dataframe
def onehot_to_signature(dataframe, Y_SIZE = Y_SIZE):
    signature_df = {'x':[], 'y':[]}
    for i in tqdm(range(len(dataframe))):
        signature_x = [onehot for onehot in dataframe[i,:-Y_SIZE]]
        signature_y = [onehot for onehot in dataframe[i,-Y_SIZE:]]
        signature_df['x'].append(flatten(signature_x))
        signature_df['y'].append(flatten(signature_y))
    return pd.DataFrame(signature_df)


dataframe = turn_to_numeric(dataframe, columns_exeptions=coordinates_columns, unique_values_dict = unique_values_dict)

# ASSIGN VULNERABILITY CLASS TO EACH BUILDING IN THE CONDENSED DATASET
vuln_df = pd.DataFrame(columns=['vulnerability_class'], index=dataframe.index)
# vuln_df.set_index(dataframe.index)

for i in dataframe.index:
    if dataframe.sez3_struttura_verticale_1[i]==1:

        if dataframe.sez3_struttura_orizzontale_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 2 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 2 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 3 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 3 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 4:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 5 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'  # SUPPOSIZIONE!! Volte con catene, senza catene, muratura cattiva qualità
        if dataframe.sez3_struttura_orizzontale_1[i] == 5 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'

    elif dataframe.sez3_struttura_verticale_1[i] == 0:

        if dataframe.sez3_struttura_orizzontale_1[i] == 0 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 0 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'C1'
        if dataframe.sez3_struttura_orizzontale_1[i] == 2 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 2 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'C1'
        if dataframe.sez3_struttura_orizzontale_1[i] == 3 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 3 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 4 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 4 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'
        if dataframe.sez3_struttura_orizzontale_1[i] == 5 and dataframe.sez3_catene_o_cordoli_1[i] == 0:
            vuln_df.vulnerability_class[i] = 'A'
        if dataframe.sez3_struttura_orizzontale_1[i] == 5 and dataframe.sez3_catene_o_cordoli_1[i] == 1:
            vuln_df.vulnerability_class[i] = 'B'

    elif dataframe.sez3_struttura_verticale_1[i] == 4 or dataframe.sez3_struttura_verticale_1[i] == 6 \
            or dataframe.sez3_struttura_verticale_1[i]==5 or dataframe.sez3_struttura_verticale_1[i] == 3:
        if dataframe.sez2_costruzioneristrutturazione1[i] == 5:
            vuln_df.vulnerability_class[i] = 'D2'
        else:
            vuln_df.vulnerability_class[i] = 'C2'

    elif dataframe.sez3_struttura_verticale_1[i] == 2:
        vuln_df.vulnerability_class[i] = 'Non identificata'

    elif dataframe.sez3_struttura_verticale_1[i] == 7:
        vuln_df.vulnerability_class[i] = 'D2'

def turn_to_strings(dataframe, columns_exeptions = [], unique_values_dict = {}):
    for column in dataframe.columns:
        if column in columns_exeptions:
            continue
        print(column)
        for idx in dataframe.index:
            val = dataframe[column][idx]
            dataframe[column][idx] = unique_values_dict[column][val]
    return dataframe

# dataframe = numeric_to_onehot(dataframe, columns_exeptions=coordinates_columns, unique_values_dict = unique_values_dict)
# dataframe = onehot_to_signature(dataframe)

dataframe_cathegorical = turn_to_strings(dataframe, columns_exeptions=coordinates_columns, unique_values_dict = unique_values_dict)

dataframe_with_vuln = pd.concat([dataframe_cathegorical, vuln_df], axis=1)
dataframe_with_vuln.to_pickle('dataframe_strutture_verticali_vuln.pkl')
dataframe_with_vuln.to_excel('dataframe_strutture_verticali_vuln.xlsx')

