# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:34:23 2023

@author: Heng2020
"""

#%%
from playsound import playsound
import os
import random
import pandas as pd

from pydub import AudioSegment
from pydub.playback import play

# import whisper 

import ffmpeg
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

# Example usage

alarm_path = "H:\D_Music\Sound Effect positive-logo-opener.mp3"
speed_factor = 0.5  # Play at 50% slower speed
################################ reload model(needs to be runed) ######################################
# spyder can't have model_base &  model_large: I have to reload everytime
#%%
def play_audio_slower(audio_path, speed_factor):
    # still not working
    # right now I use this to play audio because playsound sometimes have weird problem
    # TODO: Find way to play audio with slower speed without changing the file
    audio = AudioSegment.from_file(audio_path)
    
    # slowed_audio = audio.speedup(playback_speed=1/speed_factor)
    slow_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_factor)})
    play(slow_audio)


def get_filename(folder_path,extension = "all"):
    # also include "folder"  case
# tested small
    if extension == "all":
        out_list = [ file for file in os.listdir(folder_path) ]

    elif isinstance(extension,str):
        extension_temp = [extension]

        out_list = []

        for file in os.listdir(folder_path):
            if "." in file:
                file_extension = file.split('.')[-1]
                for each_extention in extension_temp:
                    # support when it's ".csv" or only "csv"
                    if file_extension in each_extention:
                        out_list.append(file)
            elif extension == "folder":
                out_list.append(file)


    elif isinstance(extension,list):
        out_list = []
        for file in os.listdir(folder_path):

            if "." in file:
                file_extension = file.split('.')[-1]
                for each_extention in extension:
                    # support when it's ".csv" or only "csv"
                    if file_extension in each_extention:
                        out_list.append(file)

            elif "folder" in extension:
                out_list.append(file)

        return out_list

    else:
        print("Don't support this dataype for extension: please input only string or list")
        return False

    return out_list
#%%

def format_excel(df):
    # df from excel
    # this would be different for different file

    out_df = df.iloc[1:,2:5]
    out_df.columns = out_df.iloc[0]
    out_df = out_df[1:]
    out_df.reset_index(drop=True, inplace=True)
    
    return out_df

#%%
# model_base = whisper.load_model('base')
# model_large = whisper.load_model('large')
playsound(alarm_path)
#------------------------------ reload model ------------------------------

#%%
folder_path = r"C:\Users\Heng2020\OneDrive\D_Documents\_LearnLanguages 04 BigBang PT\_BigBang PT\S06 Done\Sentence Audio"
script_path = r"C:/Users/Heng2020/OneDrive/Python NLP/NLP 03_11LabsBulk/BigBang Sentence 01_pd.xlsb"

#%%
file_path = get_filename(folder_path,[".mp3",".wav"])
# insert so that the index can start with 1
file_path.insert(0,None)
script_ori = pd.read_excel(script_path)
script = format_excel(script_ori)
#%%
start_inx = 31
end_inx = 35

mySeed = 24
#%%
# 

skip_inx = []
easy = []

# (2,'2-Jul-23')



random_inx_list = list(range(start_inx,end_inx+1))

if skip_inx:
    random_inx_list = [x for x in random_inx_list if x not in skip_inx]
else:
    random_inx_list = [x for x in random_inx_list]
#%%
print(f"Allow index is from 0 to {len(random_inx_list)-1}")
#%%
random.seed(mySeed)
random.shuffle(random_inx_list)
#%%
########################## run below recurrently ##################################
# not truely fixed my seed
chosen_inx = random_inx_list[4]
# chosen_inx = 2
print(f"Index: {chosen_inx}")
audio_path = os.path.join(folder_path,file_path[chosen_inx])
speed_factor = 1

play_audio_slower(audio_path, speed_factor)

#%%
###################### show answer
ans = script.loc[chosen_inx-1,'English']
print(ans)

ans = script.loc[chosen_inx-1,'Portuguese']
print(ans)

#%%
################################
# text_pred = model_base.transcribe(audio_path,language="pt")['text']
# print(text_pred)

# text_pred = model_large.transcribe(audio_path,language="pt")['text']
# print(text_pred)
# playsound(alarm_path)
