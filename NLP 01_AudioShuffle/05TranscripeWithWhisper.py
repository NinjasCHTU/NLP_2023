# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:17:08 2023

@author: Heng2020
"""
import torch
import whisper
from playsound import playsound

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("medium", device=device)

path01 = r"H:\D_Video\Media Learn Languese\Audio\He SELF STUDIED Chinese in 1.5 Years.wav"

alarm_path = "H:\D_Music\Sound Effect positive-logo-opener.mp3"


ans01 = model.transcribe(path01,language="zh")['text']
playsound(alarm_path)
