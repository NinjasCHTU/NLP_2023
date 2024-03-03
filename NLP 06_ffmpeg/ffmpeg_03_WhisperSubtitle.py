# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:35:19 2023

@author: Heng2020
"""

import whisper
from whisper.utils import get_writer
from playsound import playsound
from time import time
import pandas as pd
import stable_whisper

######## Not done
## stable_whisper suppose to improve timestamp syncing when creating  substitles




alarm_done_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"

ts01 = time()

input_path1 = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\Westworld S04E01 Portuguese_01.mkv"
input_path2 = r"C:\Users\Heng2020\OneDrive\D_Documents\_LearnLanguages 04 BigBang PT\_BigBang PT\S06 Done\Sentence Audio\S06E01\Normal\S06E01 Normal 001_I invited him..mp3"
output_folder  = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData"


model_fast = stable_whisper.load_faster_whisper('base')
result2 = model_fast.transcribe_stable('audio.mp3')

model = whisper.load_model("base")


result = model.transcribe(input_path1)

outputpath = r"Westworld S04E01 Portuguese_01_Whisper"

writer = get_writer("srt", str(output_folder))
writer(result,outputpath)

transcribe_df = pd.DataFrame(result['segments'])[["text","start","end","avg_logprob","no_speech_prob"]]

ts02 = time()

duration = ts02 - ts01 
print(duration)

playsound(alarm_done_path)



def write_subtitle_from_transcribe(transcribe,output_name,output_folder = None):
    
    from datetime import timedelta
    import os
    segments = transcribe['segments']
    for segment in segments:
        startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        segmentId = segment['id']+1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"
        
        srtFilename = os.path.join("SrtFiles", f"VIDEO_FILENAME.srt")
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)