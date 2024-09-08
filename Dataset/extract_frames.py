# %%
# This script is to extract frames from videos for corresponding sample pairs
import pandas as pd 
import os

# %%
# Get the path to the solid_pairs.csv file
csv_file='solid_pairs.csv'
data = pd.read_csv(csv_file)

# %%
# Set the folder where to stored the extract frames 
destDir="./Solid/frames/"
# Get the path to the folder where the videos are stored, adjust it if needed
v_dir = './Solid/videos/'


# %%
from pathlib import Path
import subprocess

# Find all entries in pairs
path_list=[]
for path in data['video_path']:
#     process the files' path
    file_name=path.replace(v_dir,'')
    path1 = Path(path)
    save_path=destDir+file_name[0:-5]+'/'
    increased_var = "%04d"

    command = f'ffmpeg -i "{path}" -vf "fps=6" "{save_path}%04d.png"'
    print(command)

    

    if path1.is_file():
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("created"+save_path)

        print("processing:")
        print(save_path)
        print("---------")

        
        subprocess.run(command, shell=True)
        



