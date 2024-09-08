# %%
import numpy as np
from PIL import Image
from transformers import VivitImageProcessor, VivitForVideoClassification
np.random.seed(0)
import pandas as pd

# %%
# Define the path to the new sample pairs CSV file, adjust it if needed
new_csv_file_path = "../Dataset/new_solid.csv"

# %%
# get the name and parent folder path in 'start_frame'

def get_file_name_and_parent_folder(file_path):
  file_name = file_path.split('/')[-1].split('.')[0]
  parent_folder = '/'.join(file_path.split('/')[:-1])
  return file_name, parent_folder


# %%
class_labels =['down','same','up']
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(f"Unique classes: {list(label2id.keys())}.")

# %%
model_ckpt = "google/vivit-b-16x2-kinetics400"
image_processor = VivitImageProcessor.from_pretrained(model_ckpt)

# %%
import torch

from torch.utils.data import Dataset


class MyCSVDataset(Dataset):
    def __init__(self, csv_file):
        # read csv_file
        self.data = pd.read_csv(csv_file)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get data item according to the index
        sample = self.data.iloc[idx]
        clip_path = sample['start_frame']
        file_name, parent_folder = get_file_name_and_parent_folder(clip_path)
        file_name = int(file_name)
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
        # pixel_values = pixel_values.squeeze(1)

        label=sample['arousal_change']
        label=label2id[label]
        label_numpy = np.array([label])
        label_tensor = torch.from_numpy(label_numpy)
        label_tensor=torch.LongTensor(label_tensor)
        inputs['label']=label_tensor

        
        return inputs


dataset = MyCSVDataset(new_csv_file_path)

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
# training
train_dataloader = DataLoader(train_dataset, batch_size=4)
# testing
test_dataloader = DataLoader(test_dataset, batch_size=1)

# %% [markdown]
# training or finetuning helper

# %%
import os
import matplotlib.pyplot as plt
# from PIL import Image

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


# %%
model = VivitForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# %% [markdown]
# Test before finetuning

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
      pixel_values = pixel_values.squeeze(1)
      label = batch.pop("label").to(device)

      outputs = model(pixel_values=pixel_values)
      logits = outputs.logits
      predicted_id = logits.argmax(-1).item()
      predicted_label = model.config.id2label[predicted_id]
      pred_result.append([idx,predicted_label])
      total += label.size(0)
      correct += (predicted_id == label).sum().item()
acc = correct / total
print("accuracy:",acc)

# %%
# save the accuarcy
with open('acc.txt', 'a') as f:
    f.write('vivit_wo_finetuning:\n')
    f.write(str(acc))
    f.write('\n')

print("successfully recorded")

# %% [markdown]
# finetuning

# %%
# lr, epoch can be changed
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
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
# save model and loss fig, name changed accordingly
model_id = './TrainedModels/finetuned_vivit_5epoch'
if not os.path.exists(model_id):
    os.makedirs(model_id)
print("model_output:", model_id)
model.save_pretrained(model_id)
plot(loss_list, model_id)

# %% [markdown]
# Test finetuned vivit

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
      pixel_values = pixel_values.squeeze(1)
      label = batch.pop("label").to(device)
      outputs = model(pixel_values=pixel_values)
      logits = outputs.logits
      predicted_id = logits.argmax(-1).item()
      predicted_label = model.config.id2label[predicted_id]
      pred_result.append([idx,predicted_label])
      total += label.size(0)
      correct += (predicted_id == label).sum().item()
acc = correct / total
print("accuracy:",acc)


# %%
# save the accuarcy
with open('acc.txt', 'a') as f:
    # change the name accordingly
    f.write('vivit_finetuned_5epochs:\n')
    f.write(str(acc))
    f.write('\n')

print("successfully recorded")


