
########################## start run 1 ################################
import pysrt
import pandas as pd
import os
from pydub import AudioSegment
from datetime import datetime, timedelta
import time
from playsound import playsound
# Parse the SRT subtitle file

video_path = r"H:\D_Video\The Ark Season 01 Portuguese\The Ark S01E01 PT.mkv"
srt_path = r"H:\D_Video\The Ark Season 01 Portuguese\Subtitles\The Ark S01E01 PT.srt"
folder_path = r"H:\D_Video\The Ark Season 01 Portuguese\Subtitles"
prefix_name = "The Ark S01E01"


# prefix_name = "BigBang_S04E01"
alarm_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"

def srt_to_df(srt_path):
    subs = pysrt.open(srt_path)
    # Initialize empty lists for storing data
    sentences = []
    start_times = []
    end_times = []

    # Extract data from each subtitle sentence
    for sub in subs:
        sentences.append(sub.text)
        start_times.append(sub.start.to_time())
        end_times.append(sub.end.to_time())

    # Create a DataFrame
    df = pd.DataFrame({
        'sentence': sentences,
        'start': start_times,
        'end': end_times
    })
    return df

# def srt_to_csv(srt_path,output_path):
#     # output should be total_path
#     df_sub = srt_to_df(srt_path)
#     # encoding='utf-8-sig' for Portuguese
#     df_sub.to_csv(output_path, encoding='utf-8-sig')
def to_ms(time_obj):
    time_obj_ms = (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + time_obj.microsecond // 1000
    return time_obj_ms

def St_NumFormat0(num, max_num=None, digit=None):
    # ChatGPT solo
    # tested
    if max_num is not None:
        num_str = str(num).zfill(len(str(max_num)))
    elif digit is not None:
        num_str = str(num).zfill(digit)
    else:
        num_str = str(num)
    # print(num_str)
    return num_str

def audio_duration(video_path):


    if isinstance(video_path,str):
        video_audio = AudioSegment.from_file(video_path)
    else:
        video_audio = video_path

    # Get the duration of the audio segment in milliseconds
    duration_ms = len(video_audio)

    # Convert the duration from milliseconds to a timedelta object
    duration = timedelta(milliseconds=duration_ms)

    # Create a dummy datetime object with a zero timestamp
    dummy_datetime = datetime(1, 1, 1, 0, 0, 0)

    # Add the duration to the dummy datetime to get the final datetime
    final_datetime = dummy_datetime + duration

    return final_datetime.time()

def print_time(duration):
    # tested
    hours = duration // 3600
    minutes = (duration % 3600) / 60
    minutes_str = "{:.2f}".format(minutes)
    seconds = duration % 60
    seconds_str = "{:.2f}".format(seconds)
    if hours < 1:
        if minutes > 1:
            # only minutes
            print(f"{minutes_str} minutes", end="\n")
        else:
            # only seconds
            print(f"{seconds_str} seconds", end="\n")
    else:
        # hours with minutes
        print(f"{hours} hour", end=" ")
        print(f"{minutes_str} minutes", end="\n")
        
####################

subs = srt_to_df(srt_path)

# TODO: define folder and loop through folder find many srt's and convert to csv at once
# TODO: write a function input is video/video path & subs/sub path
t01 = time.time()
video_audio = AudioSegment.from_file(video_path)
t02 = time.time()
t01_02 = t02-t01
print("Load video time: ", end = " ")
print_time(t01_02)

playsound(alarm_path)
# ---------------------------- run til 1 -------------------------------
########################## start run 2 ################################
t03 = time.time()
video_length = audio_duration(video_audio)
# Iterate over subtitle sentences
n = subs.shape[0]
t04 = time.time()
for i in range(n):
    start_time = subs.loc[i,'start']
    end_time = subs.loc[i,'end']
    
    if start_time > video_length:
        break

    start_time_ms = to_ms(start_time)
    end_time_ms = to_ms(end_time)

    # Extract audio segment based on timestamps
    sentence_audio = video_audio[start_time_ms:end_time_ms]
    
    num_str = St_NumFormat0(i+1,n+1)
    # Save the audio segment to a file
    audio_name = f'{prefix_name}_{num_str}.wav'
    audio_output = os.path.join(folder_path,audio_name)
    sentence_audio.export(audio_output, format='wav')
t05 = time.time()

t04_05 = t05-t04
playsound(alarm_path)
print("Split Audio time: ", end = " ")
print_time(t04_05)
print(f"Total number of sentences:{num_str}")