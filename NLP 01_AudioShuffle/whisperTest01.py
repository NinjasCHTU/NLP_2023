# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:29:13 2023

@author: Heng2020
"""
# I workssss finally
import whisper 

import ffmpeg
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

def play_audio_slower(audio_path, speed_factor):
    # still not working
    # TODO: Find way to play audio with slower speed without changing the file
    audio = AudioSegment.from_file(audio_path)
    
    # slowed_audio = audio.speedup(playback_speed=1/speed_factor)
    slow_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_factor)})
    play(slow_audio)

model = whisper.load_model('base')
audio_path = r"C:\Users\Heng2020\OneDrive\Icon File Image\whisper_test_01.wav"

WW_S04E01_008 = r"C:\Users\Heng2020\OneDrive\Python NLP\OutputData\Westworld S04E01\Westworld_S04E01_008.wav"
WW_S04E01_010 = r"C:\Users\Heng2020\OneDrive\Python NLP\OutputData\Westworld S04E01\Westworld_S04E01_010.wav"

path01 = r"H:\D_Music\2022 01 单依纯  永不失联的爱.mp3"

result = model.transcribe(audio_path)
text = result['text']


S04E01_008_text = model.transcribe(WW_S04E01_008)['text']

S04E01_010_text = model.transcribe(WW_S04E01_010,fp16=False)['text']
text01 = model.transcribe(path01,fp16=False)['text']

play_audio_slower(WW_S04E01_010,1)
