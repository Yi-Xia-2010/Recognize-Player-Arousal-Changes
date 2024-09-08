# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
from PIL import Image
import pandas as pd
random.seed(0)
np.random.seed(0)
from transformers import VivitImageProcessor
# %% [markdown]
# Prepare dataset

# %%
model_ckpt = "google/vivit-b-16x2-kinetics400"
image_processor = VivitImageProcessor.from_pretrained(model_ckpt)

# %%
class_labels =['down','same','up']
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(f"Unique classes: {list(label2id.keys())}.")

# %%
def get_file_name_and_parent_folder(file_path):
  file_name = file_path.split('/')[-1].split('.')[0]
  parent_folder = '/'.join(file_path.split('/')[:-1])
  video_name = file_path.split('/')[-2]
  player_id, session_id=video_name.split('_solid_')
  return file_name, parent_folder, player_id, session_id

# %%
import torch

from torch.utils.data import Dataset


class MyCSVDataset(Dataset):
    def __init__(self, csv_file, csv_file_2):
        
        self.data = pd.read_csv(csv_file)
        self.gf = pd.read_csv(csv_file_2)
        # revome control, label, and str features in clean_data
        self.gf = self.gf.drop(columns=['[control]genre','[control]game', '[control]time_index','[output]arousal','[string]key_presses','[string]player_aim_target','[string]bot_damaged_by'],)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        gf = self.gf
        clip_path = sample['start_frame']
        file_name, parent_folder, player_id, session_id= get_file_name_and_parent_folder(clip_path)
        file_name = int(file_name)

        start=sample['start_time']
        start = int(start)*4
        end = start+24

        # transform to tensor
        game_vactor = gf[(gf['[control]player_id']==player_id)&(gf['[control]session_id']==session_id)]
        game_vactor= game_vactor.drop(columns=['[control]player_id', '[control]session_id'],)
        
        pd.set_option('future.no_silent_downcasting', True)
        game_vactor= game_vactor.fillna(0).infer_objects(copy=False)
        game_vactor = game_vactor.iloc[start:end]
        game_array = np.array(game_vactor.values)
        game_tensor = torch.from_numpy(game_array)
        game_tensor = game_tensor.float()

        frames=[]
        for i in range(32):
            if (file_name<10):
                frame_path = parent_folder + "/000" + str(file_name)+ ".png"
            elif (10<=file_name<100):
                frame_path = parent_folder + "/00" + str(file_name)+ ".png"
            else:
                frame_path = parent_folder + "/0" + str(file_name)+ ".png"

            # the frames path, adjust it if needed
            frame_path = "../Dataset" + frame_path[1:]
          
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
            file_name += 1

        inputs = image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values']
        pixel_values = pixel_values.squeeze(0)
        inputs['pixel_values'] = pixel_values

        label=sample['arousal_change']
        label=label2id[label]
        label_numpy = np.array([label])
        label_tensor = torch.from_numpy(label_numpy)
        label_tensor=torch.LongTensor(label_tensor)
        
        inputs['label']=label_tensor
        inputs['game_tensor'] = game_tensor 

        return inputs

# path to the helper file and clean_data file, adjust it if needed
csv_file = '../Dataset/new_solid.csv'
csv_file_2 = "../Dataset/clean_data.csv"
dataset = MyCSVDataset(csv_file,csv_file_2)

# %%
torch.manual_seed(0)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [10240, 2560]
)

# %% [markdown]
# Traning

# %%
# transform data into tensors

game_tensors = []
labels = []

for data in train_dataset:
    game_tensor = data['game_tensor'].numpy()
    label = data['label'].numpy()
    
    game_tensors.append(game_tensor)
    labels.append(label)

game_tensors_np = np.array(game_tensors)
labels_np = np.array(labels)

print(f'Game Tensors Shape: {game_tensors_np.shape}')
print(f'Labels Shape: {labels_np.shape}')

game_tensors_np=game_tensors_np.reshape(game_tensors_np.shape[0], -1)

# %%
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(game_tensors_np, labels_np)

# %% [markdown]
# Test

# %%
test_game_tensors = []
test_labels = []

for data in test_dataset:
    test_game_tensor = data['game_tensor'].numpy()
    test_label = data['label'].numpy()
    
    test_game_tensors.append(test_game_tensor)
    test_labels.append(test_label)

test_game_tensors_np = np.array(test_game_tensors)
test_labels_np = np.array(test_labels)
test_game_tensors_np=test_game_tensors_np.reshape(test_game_tensors_np.shape[0], -1)

# %%
test_pred = clf.predict(test_game_tensors_np)

# %%
acc = accuracy_score(test_labels, test_pred)
print(f'Accuracy: {acc}')

# %%
with open('acc.txt', 'a') as f:
    f.write('acc_random_forest:\n')
    f.write(str(acc))
    f.write('\n')

print("record ")


