# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:51:40 2023

@author: Heng2020
"""
# nltk only supports English

import nltk
import pandas as pd
# nltk.download('omw-1.4')
from playsound import playsound
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy

language = 'portuguese'
alarm_path = "H:\D_Music\Sound Effect positive-logo-opener.mp3"
playsound(alarm_path)


# Specify the path to the .txt file
file_path = 'C:/Users/Heng2020/OneDrive/Python NLP/NLP 02/Westworld S04E01 Portuguese.txt'


with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    
tokens = word_tokenize(text)

lower_tokens = [text.lower() for text in tokens ]

word_count = Counter(lower_tokens)

stop_words = list(stopwords.words('portuguese'))
print(list(stopwords.words('portuguese')))

alpha_only = [t for t in lower_tokens if t.isalpha()]
no_stops = [t for t in alpha_only if t not in stop_words]

# WordNetLemmatizer in nltk only supports English!!!
wordnet_lemmatizer = WordNetLemmatizer()

lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
word_count02 = Counter(lemmatized)
word_count03 = Counter(alpha_only)

# small:    'pt_core_news_sm'
# medium:   'pt_core_news_md'
# large:    'pt_core_news_lg'

nlp = spacy.load('pt_core_news_sm')
doc = nlp(text)
sentences =  [sent.text.strip() for sent in doc.sents if sent.text.strip()]

token_dicts = [{'vocab': token.text.lower(), 'function': token.pos_, 'infinitive':token.lemma_} for token in doc if not token.is_stop ]
text_df = pd.DataFrame(token_dicts)
word_count = Counter(text_df['vocab'])

part_of_speech = text_df['function'].unique().tolist()
part_of_speech = ['ADP', 'VERB', 'PUNCT', 'SPACE', 'NOUN', 'PRON', 'ADJ', 'ADV', 'AUX', 'PROPN', 'NUM', 'SCONJ', 'DET', 'INTJ', 'X']
stop_part = ['']
text_list = [token.text for token in doc]

entity_dict = [{'word':ent.text, 'label':ent.label_} for ent in doc.ents ]
entity = pd.DataFrame(entity_dict)
for ent in doc.ents:
    print(ent.text, ent.label_)

# Use to explain the abbreviation
# spacy.explain('PER')



