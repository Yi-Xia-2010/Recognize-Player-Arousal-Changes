# %%
# This script is to generate a helper file which help to transform the Solid data into sample pairs
import pandas as pd
import csv
import os
import glob

# %%
# create a csv file with columns name as video_path,start,end,label.  Adjust the path if needed
# This is to calculate the arousal value, generate the label for each time windows, and store the info about each sample pairs. 
# This file will help prepare the dataset in later parts
with open('solid_pairs.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the column names, 'start' is the start time, 'end' is the end time
    writer.writerow(['video_path', 'start', 'end', 'label'])






# %%

# Get the path to the folder where the videos are stored, adjust it if needed
folder_path = './Solid/videos'

# Find all files in the folder
files = glob.glob(os.path.join(folder_path, '*'))

# Get the names of the files
file_names = [os.path.basename(file) for file in files]



# %%
gameplay_info = pd.read_csv('clean_data.csv')

# %%
# This is to generate entries about 3-second time windows in Solid from the clean_data.csv
down = 0
up = 0
total = 0
for file in file_names:
  file_path = str(folder_path)+'/'+str(file)
  print("processing:",file)
  file_name = file[0:-5]
  id_list = file_name.split('_solid_')

  # Find the entries with Solid game sessions
  df1 = gameplay_info.loc[(gameplay_info['[control]player_id']==id_list[0])&(gameplay_info['[control]session_id']==id_list[1])]
  
# Four data per 1 second, interval step is to 4, i represents data index, we want to get the first index in each second
  for i in range(0,len(df1),4):
    # If there is no overlap, the first data index is i at the beginning of every 6 seconds(one sample pair) and the last data index is i+23;
    # i+24 is then the index of the first data of the next 6 seconds. 
    if ((i+23)<len(df1)):
      # Here, To calculate start time and end time of each time window
      start = float(i)
      start = start/4.0
      end = float(i+24)
      end = end/4.0

      # Calculate the arousal value of the corresponding time window based on the time and index.
      df2 = df1.iloc[i:i+24]
      arousal_1 = 0
      arousal_2 = 0

      # The index range for the first 3 seconds is 0-11, including 11, for a total of 12 entries of data
      for j in range(0,12):
        df3 = df2.iloc[j]
        arousal = df3['[output]arousal']
        arousal_1 = arousal_1+arousal
      # average
      arousal_1 = arousal_1/12
      #The index range for the last 3 seconds is 12-23
      for k in range(12,24):
        df3 = df2.iloc[k]
        arousal = df3['[output]arousal']
        arousal_2 = arousal_2+arousal
      arousal_2 = arousal_2/12
      # keep labels simple, arousal unchanging-->"same", arousal decreasing-->"down", arousal increasing-->"up"
      arousal_change = "same"
      if(arousal_1>arousal_2):
        arousal_change = "down"
        down = down+1
      if(arousal_1<arousal_2):
        arousal_change = "up"
        up = up+1
      total = total + 1

      with open('solid_pairs.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file_path, start, end, arousal_change])

  # As two pairs cannot extract corresponding frames from videos, then removed in later process
  # print("down:",down,"; up:",up, "total:", total)









# %%



