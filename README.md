# Recognizing-Player-Arousal-Changes-in-Games
This is the repository containing the supplement files for the paper "Knowing Arousal Changes without Seeing You: Recognizing the Player’s Arousal Changes in the Game through Game Footage Videos and Game Context Features"  

Please note:    
    1. The default relative position (commented in scripts) has been set in our code, please adjust it yourself if needed.   
    2. Python version >= 3.11.7  


## Dataset
This project uses a dataset from The Arousal Video Game AnnotatIoN (AGAIN) Dataset, which is licensed under the MIT License. Please refer to the LICENSE file in the Solid folder for more information.  

As the files in the dataset are too large to upload to GitHub, you can check and download the dataset here by following their instructions: https://www.again.institutedigitalgames.com/   

In this project, we used their clean_data.csv file and the gameplay videos named [ParticipantID]_[Solid]_[SessionID].webm, which correspond to the entries in the Solid game in the clean_data.csv file. You can download just these files.

This folder also contains the data preprocessing scripts we used.

The solid_pairs.csv is the helper file generated by the prepare_sample_pair.py.  

The new_solid.csv is the updated helper file generated by the update_paris.py.

## Models
This folder contains the scripts for training, finetuning, and evaluating the models.  

The acc.txt file is the file containing the accuracies of the models on the test set. 

Checkpoints of the Trained Models:
| Model | Link |
|----------|----------|
| Finetuned ViViT 1(10 epochs) | [download](https://drive.google.com/drive/folders/1VSUL2-XHr5sp_mtwJcTGwJoa8KNb-OnY?usp=drive_link)| 
| Finetuned ViViT 2(15 epochs) | [download](https://drive.google.com/drive/folders/14x9Qb2qxn5HpwJLQPoRkjNAhbo8MnSGZ?usp=drive_link) |
| MLP GF encoder | [download](https://drive.google.com/file/d/1rT1j6ugatiFsV0I52X3Cen2BEsqVhHoh/view?usp=drive_link) |
| Finetuned ViViT 1 + MLP GF encoder| [download](https://drive.google.com/file/d/1P4szSf3wOKlPBnDOV9qxKjfZlvXOh8X9/view?usp=drive_link)  |
| Finetuned ViViT 2 + MLP GF encoder| [download](https://drive.google.com/file/d/1Ya_u_72jQxs63orvLeVjxYNtWXmx2Cn_/view?usp=drive_link)  |


## Installation and Usage
0. Create a virtual environment(if needed)    

    We used conda to build a virtual environment and help manage the dependencies:
    ```bash
    conda create -n myenv python=3.11  
    conda activate myenv 
    ```
    
1. Install the requirements:
    ```bash
    conda install --file conda_requirements.txt
    ```
    Or use:
    ```bash
    pip install -r requirements.txt
    ```
    We used ffmpeg to extract the frames from videos. You need to download and install it following the instruction here:
    https://ffmpeg.org/download.html
   
    In conda environment, you can use this commond to install ffmpeg:
    ```bash
    conda install -c conda-forge ffmpeg
    ```

2. Download the dataset  
   See the Dataset part.    
   Put the dataset under the Dataset folder.   
   The videos from the AGAIN dataset can be put in videos folder.  
   Scripts will generate helper files and put them under Dataset folder. And Script also extract Frames and put them into frames folder.    
   The clean_data.csv is from the dataset.
   The path in our default setting:  
   Dataset/ Solid/ videos/   
   Dataset/ Solid/ frames/       
   Dataset/ clean_data.csv


3. Run scripts:  
   3.1 Data preprocessing:  
   Go into Dataset folder, run the python scripts one by one, You can also choose to run the jupyter notebook files accordingly.
    
   This script is to generate a helper file which help to transform the Solid data into sample pairs :    
   ```bash
   python prepare_sample_pair.py
   ```

   This script is to extract frames from videos for corresponding sample pairs:  
   ```bash
   python extract_frames.py
   ```

   This script is to generate a new csv helper file containing the pairs' info:
   ```bash
   python update_pairs.py
   ```
   
   3.2 Modeling：  
   Go into Models folder, run the scripts:

   This script is to train and evaluate the random forest classifier(baseline) on our preprocessed dataset:
   ```bash
   python random_forest.py
   ```

   This script is to finetune the vivit-b-16x2-kinetics400 model and evaluate the finetuned models:
   ```bash
   python vivit_finetuned.py
   ```

   This script contains the code for experimenting with our methods. As it includes the process of training mlp and vivit from scratch,
   ```bash
   python our_modeling.py
   ```
   As our modeling used the finetuned vivit models, vivit_finetuned.py should be run before our_modeling.py. As it includes the process of training mlp model and vivit model from scratch, it is recommended that you run different parts of the jupyter notebook file separately.

  









