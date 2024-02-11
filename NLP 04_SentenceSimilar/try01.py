# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 06:59:58 2023

@author: Heng2020
"""
####### !!! similar score doesn't seem to work well as compared to
# GPT_SimilarScore
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import unittest
import xlwings as xw
from playsound import playsound

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
import lib02_dataframe as f1

# declare model outside bc it will take a lot of time to load the model
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"

def similar_meaning(sen1, sen2,load_model = False):
# load_model = True will reload model everytime I call the function
# which will slow the code down
    global model
    if load_model:
        model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    
    if not isinstance(sen2, list):
        sen2_list = [sen2]
    else:
        sen2_list = sen2
    
    # Encode sen1
    emb1 = model.encode(sen1, convert_to_tensor=True)
    
    # Encode all sentences in sen2_list
    emb2_list = model.encode(sen2_list, convert_to_tensor=True)
    
    # Compute cosine similarity between emb1 and each of emb2_list
    cosine_similarities = util.pytorch_cos_sim(emb1, emb2_list)[0]
    
    # Convert the tensor to a list of Python floats
    similar_scores = [int(score * 100) for score in cosine_similarities]
    
    if len(similar_scores) == 1:
        return similar_scores[0]

    return similar_scores

def St_SplitSentence(text, delimiter, inplace=True):
    if not inplace:
        text = text.copy()
        
    i = 0
    while i < len(text):
        text[i] = text[i].strip()  # Trim the spaces at both ends
        if delimiter in text[i]:
            # Split the string using the delimiter
            split_strings = text[i].split(delimiter)
            
            # Remove the original string from the list
            del text[i]
            
            # Insert the split strings back into the original list at the same position
            for split_str in reversed(split_strings):
                split_str = split_str.strip()  # Remove leading and trailing spaces
                if split_str:  # Only add non-empty strings
                    text.insert(i, split_str)
        else:
            i += 1  # Only increment if no split occurred to handle new inserted strings
            
    return text if not inplace else None

# Function to remove elements that start with a specific character (in this case "♪")
def remove_from_list(lst, char="♪"):
    # can generalize more to 
    # remove_from_list(lst,start_with = None,end_with = None,logic = "or")
    return [element for element in lst if not str(element).startswith(char)]

def similar_text_matrix(sentences,translations,n_row_down = 10,n_row_up=0):
    # dependency: similar_meaning
    output_score = []
    for i,sentence in enumerate(sentences):
        start_inx = max(0,i-n_row_up)
        end_inx = min(len(sentences),i+n_row_down)
        to_compare = translations[start_inx:end_inx]
        score_list = similar_meaning(sentence,to_compare)
        output_score.append(score_list)
    return output_score

def similar_text_df(sentences,translations,score_matrix):
    # score_matrix is 2d list
    index_list = []
    similar_sentence = []
    
    for i, score_list in enumerate(score_matrix):
        # well what if the start_inx is negative from the start?
        
        if isinstance(score_list, list):
            max_score = max(score_list)
            max_position = score_list.index(max_score)
        else:
            # in case score_list is just a number
            max_score = score_list
            max_position = 0
        
        most_similar = translations[i+max_position]
        
        
        index_list.append(max_position)
        similar_sentence.append(most_similar)
        
    out_df = pd.DataFrame({'sentence':sentences,
                           'most_similar':similar_sentence,
                           'chosen_index':index_list
                           })
    return out_df
        
        
    
    
    

start_adr = "j2"
ws_name = "EP2_2"


sen1 = "Então você se vê mesmo casando com o Sheldon um dia?"
sen2 = "So, you actually see you and Sheldon getting married someday?"



folder = r"C:\Users\Heng2020\OneDrive\D_Documents\_LearnLanguages 04 BigBang PT\_BigBang PT\S06 Done"
folder2 = r"C:\Users\Heng2020\OneDrive\D_Documents\_LearnLanguages 04 BigBang PT"
output_name = "BigBang Sentence 01.xlsm"


PT_script = "BigBang S06E02 PT.xlsx"
EN_script = "BigBang S06E02 EN.xlsx"

PT_path = folder + "\\" + PT_script
EN_path = folder + "\\" + EN_script
output_path = folder2 + "\\" + output_name

wb = xw.Book(output_path)
ws_data_name = "EP2_2"
ws_input = wb.sheets[ws_data_name]

PT_list_ori = ws_input.range('B2:B94').value
EN_list_ori = ws_input.range('C2:C94').value

# PT_df = pd.read_excel(PT_path)
# EN_df = pd.read_excel(EN_path)

# PT_list_ori = PT_df['sentence'].to_list()
# EN_list_ori = EN_df['sentence'].to_list()

PT_list = St_SplitSentence(PT_list_ori,"- ",inplace=False)
 # (remove "♪")
EN_list = remove_from_list(EN_list_ori)

# num01 = [[11,21],[22,23]]
PT_list.pop(1)

sample_EN = EN_list[:50]
sample_PT = PT_list[:50]

# to reduce unwated row temporally

#######################


ws_names = [sheet.name for sheet in wb.sheets]

# Open or add the sheet 'output_syc'
if ws_name in ws_names:
    ws = wb.sheets[ws_name]
else:
    ws = wb.sheets.add(ws_name)

# ws.range(start_adr).value = num01

print(similar_meaning(sen1, sen2))
similar_score1 = similar_text_matrix(sample_EN,sample_PT)
playsound(alarm_path)


similar_score2 = similar_score1[:39]
ws.range(start_adr).value = similar_score2


text_df1 = similar_text_df(sample_PT,sample_EN,similar_score1)



