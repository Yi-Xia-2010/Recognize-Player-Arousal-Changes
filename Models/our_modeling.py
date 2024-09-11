# %%
import numpy as np
from PIL import Image
import pandas as pd
import os

# %%
import torch
from transformers import VivitImageProcessor, VivitForVideoClassification
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)


# %% [markdown]
# # Prepare dataset

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
  file_path = os.path.normpath(file_path)
  file_paths = file_path.split(os.sep)
  file_name, file_extension = os.path.splitext(os.path.basename(file_path))
  parent_folder = os.path.dirname(file_path)
  video_name = file_paths[-2]
  player_id, session_id=video_name.split('_solid_')
  return file_name, parent_folder, player_id, session_id

# %%
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
            frame_path = "../Dataset/" + frame_path
            frame_path = os.path.normpath(frame_path)
          
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

# %% [markdown]
# Split train and test
# 

# %%
torch.manual_seed(0)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [10240, 2560]
)

# %% [markdown]
# dataloader

# %%
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4)

# %%
test_dataloader = DataLoader(test_dataset, batch_size=1)

# %% [markdown]
# Training or finetuning helper

# %%
import os
import matplotlib.pyplot as plt


# plot helper
def plot(loss_list, output_path):
    plt.figure(figsize=(10,5))

    freqs = [i for i in range(len(loss_list))]
    # Plotting training loss curves
    plt.plot(freqs, loss_list, color='#e4007f', label="train/loss curve")

    # Plotting axes and legends
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.savefig(output_path+'/pytorch_vivit_loss_curve.png')
    # plt.show()

# %% [markdown]
# # Training or finetuning

# %%
from torch import nn

# load finetuned vivit.(finetuned_vivit_10epochs (finetuned_vivit_1) or finetuned_vivit_15epochs(finetuned_vivit_2))
# adjust the model path if needed

video_encoder = VivitForVideoClassification.from_pretrained(
    "TrainedModels/finetuned_vivit_15epochs",
    label2id = label2id,
    id2label = id2label,
    ignore_mismatched_sizes = True,
)

# %%

# Freeze the paremeters in finetuned vivit.

for param in video_encoder.parameters():
    param.requires_grad = False

# %% [markdown]
# ### MLP GF Encoder

# %%
# Defined the MLP GF encoder
class GfEncoder(nn.Module):
    def __init__(self):
        super(GfEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(24*112, 1024),  # input layer
            nn.ReLU(),  
            nn.Linear(1024, 512),  # hidden layer
            nn.ReLU(),  
            nn.Linear(512, 768),  # output layer
        )
        
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # adjust shape
        x = self.layers(x)
        return x.view(x.size(0), -1)  # Adjust the shape of the output tensor to [batch_size, 768]
    

gf_encoder = GfEncoder()

# %% [markdown]
# ### Overall contrastive model

# %%
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, gf_model, video_model):
        super(ContrastiveModel, self).__init__()
        self.gf_model = gf_model
        self.video_model = video_model

        # learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]))

        

     

    def forward(self, gf_inputs, video_inputs):
        gf_outputs = self.gf_model(gf_inputs)  # game context feature representation output from GF encoder 
        video_outputs = self.video_model(video_inputs,output_hidden_states=True)  # game footage videos representation output from video encoder 
        video_outputs = video_outputs.hidden_states[-1][:, 0, :]

        # normalize
        gf_outputs = gf_outputs / gf_outputs.norm(dim=1, keepdim=True)
        video_outputs = video_outputs / video_outputs.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()  
        logits_per_video = logit_scale * video_outputs @ gf_outputs.t()  # similarity
        logits_per_gf = logits_per_video.t()  

    
        return gf_outputs, video_outputs ,logits_per_video, logits_per_gf


# %%
model=ContrastiveModel(gf_encoder,video_encoder)

# %%
# lr, epoch can be changed
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size=len(train_dataset)
model.to(device)
model.train()
loss_list = []
for epoch in range(11):
    print("Epoch:", epoch)
    sum_loss_list = []
    for idx, batch in enumerate(train_dataloader):

        pixel_values = batch.pop("pixel_values").to(device)
        gf = batch.pop("game_tensor").to(device)

        gf_outputs, video_outputs, logits_per_video, logits_per_gf = model(gf_inputs= gf,video_inputs=pixel_values)

        # generate contrastive learning label
        labels = torch.arange(logits_per_video.size(0), device=logits_per_video.device)

        # calculate loss
        loss = (
            F.cross_entropy(logits_per_video, labels) +
            F.cross_entropy(logits_per_gf, labels)
        ) / 2


        print("Epoch:",epoch," , idx:",idx," , Loss:", loss.item())
        sum_loss_list.append(float(loss.item()))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
    print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
    loss_list.append(float(avg_sum_loss))

# %%
# the path to save model,adjust if needed
model_id ="./TrainedModels/contrastive_15epoch"

if not os.path.exists(model_id):
    os.makedirs(model_id)
model_file = 'model.pt'
print("model_output:", model_id)
torch.save(model, os.path.join(model_id, model_file))
plot(loss_list, model_id)

# %% [markdown]
# add classifier

# %%
# load trained model
contrastive_model = torch.load(os.path.join(model_id, 'model.pt'))

# Freeze the parameters of the model
for param in contrastive_model.parameters():
    param.requires_grad = False

# %%
# model with classifier layer
class ContrastiveForClassification(nn.Module):
    def __init__(self, ContrastiveModel,num_classes):
        super(ContrastiveForClassification, self).__init__()
        self.ContrastiveModel = ContrastiveModel
        self.classifier = nn.Linear(in_features=1536, out_features=num_classes)
    

    def forward(self, gf_inputs, video_inputs):
        # get the representations from GF encoder and video encoder after contrastive learning 
        gf_outputs, video_outputs, logits_per_video, logits_per_gf = self.ContrastiveModel(gf_inputs, video_inputs)

        # concatenate gf_outputs and video_outputs 
        x = torch.cat([gf_outputs, video_outputs], dim=1)
        x = self.classifier(x)      

        return x

# %%

model = ContrastiveForClassification(contrastive_model,3)

# %%
# train classifier layer, lr and epoch can be changed
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size=len(train_dataset)
model.to(device)
model.train()
loss_list = []
for epoch in range(11):
    print("Epoch:", epoch)
    sum_loss_list = []
    for idx, batch in enumerate(train_dataloader):

        pixel_values = batch.pop("pixel_values").to(device)        
        gf = batch.pop("game_tensor").to(device)
        label = batch.pop("label").to(device)

        outputs = model(gf_inputs= gf,video_inputs=pixel_values)
        label = label.squeeze(1)

        loss = criterion(outputs, label)
        print("Epoch:",epoch," , idx:",idx," , Loss:", loss.item())
        sum_loss_list.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
    print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
    loss_list.append(float(avg_sum_loss))


# %%
output_path ="./TrainedModels/contrastive_15epochs_classification"
model_id = output_path

if not os.path.exists(model_id):
    os.makedirs(model_id)
model_file = 'model.pt'
print("model_output:", model_id)
torch.save(model, os.path.join(model_id, model_file))
plot(loss_list, model_id)

# %%
# test
correct = 0
total = 0
pred_result = []
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):
      pixel_values = batch.pop("pixel_values").to(device)
      gf = batch.pop("game_tensor").to(device)

      label = batch.pop("label").to(device)
      outputs = model(gf_inputs= gf,video_inputs=pixel_values)
      label = label.squeeze(1)

      predicted_id = outputs.argmax(-1).item()
      pred_result.append([idx,predicted_id])
      total += label.size(0)
      correct += (predicted_id == label).sum().item()
acc = correct / total
print("accuracy:",acc)


# %%
with open('acc.txt', 'a') as f:
    f.write('acc_contrastive_15epoch_classification:\n')
    f.write(str(acc))
    f.write('\n')

print("successfully recorded")

# %% [markdown]
# ### Train MLP GF encoder (directly train on game features)

# %%
class GfEncoder_2(nn.Module):
    def __init__(self):
        super(GfEncoder_2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(24*112, 1024),  
            nn.ReLU(), 
            nn.Linear(1024, 512),  
            nn.ReLU(),  
            nn.Linear(512, 768),  
        )
        self.classifier = nn.Linear(in_features=768, out_features=3)
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  
        representation = self.layers(x)
        x = self.classifier(representation)
        return representation, x  
    



# %%
model = GfEncoder_2()

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size=len(train_dataset)
model.to(device)
model.train()
loss_list = []
for epoch in range(11):
    print("Epoch:", epoch)
    sum_loss_list = []
    for idx, batch in enumerate(train_dataloader):
      
        gf = batch.pop("game_tensor").to(device)
        label = batch.pop("label").to(device)

        representations, outputs = model(gf)
        label = label.squeeze(1)

        loss = criterion(outputs, label)

        print("Epoch:",epoch," , idx:",idx," , Loss:", loss.item())
        sum_loss_list.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
    print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
    loss_list.append(float(avg_sum_loss))

# %%
output_path ="./TrainedModels/MLP_classification_supervise"
model_id = output_path

if not os.path.exists(model_id):
    os.makedirs(model_id)
model_file = 'model.pt'
print("model_output:", model_id)
torch.save(model, os.path.join(model_id, model_file))
plot(loss_list, model_id)

# %%
correct = 0
total = 0
pred_result = []
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):
     
      gf = batch.pop("game_tensor").to(device)

      label = batch.pop("label").to(device)
      representations, outputs = model(gf)
      label = label.squeeze(1)    
      print('output shape: ',outputs.size())
      predicted_id = outputs.argmax(-1).item()
      pred_result.append([idx,predicted_id])
      total += label.size(0)
      correct += (predicted_id == label).sum().item()
acc = correct / total
print("accuracy:",acc)

# %%
with open('acc.txt', 'a') as f:
    f.write('MLP_classification_supervise:\n')
    f.write(str(acc))
    f.write('\n')

print("successfully recorded")

# %% [markdown]
# ### Only use representations from GF encoder trained by using contrastive learning

# %%
contrastive_model = torch.load(os.path.join('./TrainedModels/contrastive_15epoch', 'model.pt'))

for param in contrastive_model.parameters():
    param.requires_grad = False

# %%
class ContrastiveMLPForClassification(nn.Module):
    def __init__(self, ContrastiveModel,num_classes):
        super(ContrastiveMLPForClassification, self).__init__()
        self.ContrastiveModel = ContrastiveModel
        self.classifier = nn.Linear(in_features=768, out_features=num_classes)
    

    def forward(self, gf_inputs, video_inputs):
        
        gf_outputs, video_outputs, logits_per_video, logits_per_gf = self.ContrastiveModel(gf_inputs, video_inputs)

        x = self.classifier(gf_outputs)     

        return x

# %%
model = ContrastiveMLPForClassification(contrastive_model,3)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size=len(train_dataset)
model.to(device)
model.train()
loss_list = []
for epoch in range(6):
    print("Epoch:", epoch)
    sum_loss_list = []
    for idx, batch in enumerate(train_dataloader):

        pixel_values = batch.pop("pixel_values").to(device)        
        gf = batch.pop("game_tensor").to(device)
        label = batch.pop("label").to(device)

        outputs = model(gf_inputs=gf, video_inputs=pixel_values)
        label = label.squeeze(1)

        loss = criterion(outputs, label)

        print("Epoch:",epoch," , idx:",idx," , Loss:", loss.item())
        sum_loss_list.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
    print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
    loss_list.append(float(avg_sum_loss))

# %%
output_path ="./TrainedModels/Contrastive_15epoch_MLP_only_classification"
model_id = output_path

if not os.path.exists(model_id):
    os.makedirs(model_id)
model_file = 'model.pt'
print("model_output:", model_id)
torch.save(model, os.path.join(model_id, model_file))
plot(loss_list, model_id)

# %%
correct = 0
total = 0
pred_result = []
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):
      pixel_values = batch.pop("pixel_values").to(device)        
      gf = batch.pop("game_tensor").to(device)

      label = batch.pop("label").to(device)
      outputs = model(gf_inputs=gf, video_inputs=pixel_values)
      label = label.squeeze(1)
      
      print('output: ',outputs)
      print('label: ',label)
      
      predicted_id = outputs.argmax(-1).item()
      pred_result.append([idx,predicted_id])
      total += label.size(0)
      correct += (predicted_id == label).sum().item()
acc = correct / total
print("accuracy:",acc)

# %%
with open('acc.txt', 'a') as f:
    f.write('Contrastive_15epoch_MLP_only_classification:\n')
    f.write(str(acc))
    f.write('\n')

print("successfully recorded")

# %% [markdown]
# ### Extract GF encoder from trained contrastive model

# %%
gf_encoder_part = contrastive_model.ContrastiveModel.gf_model

# %%
gf_classifier = contrastive_model.classifier

# %%
class ContrastiveMLPForClassification_2(nn.Module):
    def __init__(self, encoder,classifier):
        super(ContrastiveMLPForClassification_2, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
    

    def forward(self, gf_inputs):
        
        gf_inputs = gf_inputs.view(gf_inputs.size(0), -1) 
        gf_outputs= self.encoder(gf_inputs)

        x = self.classifier(gf_outputs)
      
        return x

# %%
model = ContrastiveMLPForClassification_2(gf_encoder_part,gf_classifier)

# %%
correct = 0
total = 0
pred_result = []
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):      
      gf = batch.pop("game_tensor").to(device)

      label = batch.pop("label").to(device)
      outputs = model(gf_inputs=gf)
      label = label.squeeze(1)   

      predicted_id = outputs.argmax(-1).item()
      pred_result.append([idx,predicted_id])
      total += label.size(0)
      correct += (predicted_id == label).sum().item()
acc = correct / total
print("accuracy:",acc)

# %%
with open('acc.txt', 'a') as f:
    f.write('Contrastive_15epoch_extracted_MLP_classification:\n')
    f.write(str(acc))
    f.write('\n')

print("successfully recorded")

# %%
# Training extract GF encoder, loss is extremly high, stop in the half.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


criterion = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size=len(train_dataset)
model.to(device)
model.train()
loss_list = []
for epoch in range(6):
    print("Epoch:", epoch)
    sum_loss_list = []
    for idx, batch in enumerate(train_dataloader):
      
        gf = batch.pop("game_tensor").to(device)
        label = batch.pop("label").to(device)

        outputs = model(gf_inputs=gf)
        label = label.squeeze(1)

        loss = criterion(outputs, label)

        print("Epoch:",epoch," , idx:",idx," , Loss:", loss.item())
        sum_loss_list.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
    print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
    loss_list.append(float(avg_sum_loss))

# %%
output_path ="./TrainedModels/Contrastive_15epoch_extracted_MLP_classification"
model_id = output_path

if not os.path.exists(model_id):
    os.makedirs(model_id)
model_file = 'model.pt'
print("model_output:", model_id)
torch.save(model, os.path.join(model_id, model_file))


# %%


# %% [markdown]
# ### Direct train vivit from scrach

# %%
from transformers import VivitConfig, VivitForVideoClassification

# Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
configuration = VivitConfig(num_labels=3)

# Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
model = VivitForVideoClassification(configuration)


# %%
# loss is extremly high, stop in the half.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size=len(train_dataset)
model.to(device)
model.train()
loss_list = []
for epoch in range(11):
    print("Epoch:", epoch)
    sum_loss_list = []
    for idx, batch in enumerate(train_dataloader):


        pixel_values = batch.pop("pixel_values").to(device)
        pixel_values = pixel_values.squeeze(1)
        label = batch.pop("label").to(device)

        outputs = model(pixel_values=pixel_values,labels=label)
        loss = outputs.loss
        print("Epoch:",epoch," , idx:",idx," , Loss:", loss.item())
        sum_loss_list.append(float(loss.item()))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
    print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
    loss_list.append(float(avg_sum_loss))

# %%
output_path ="./TrainedModels/vivit_fromscrach"
model_id = output_path

if not os.path.exists(model_id):
    os.makedirs(model_id)
model_file = 'model.pt'
print("model_output:", model_id)
torch.save(model, os.path.join(model_id, model_file))


