# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:56:58 2023

@author: Heng2020
"""

from pathlib import Path
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from playsound import playsound

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
import lib02_pandas as f 

sys.path.append(r"C:/Users/Heng2020/OneDrive/W_Documents/DSC 2023/prepare/code")
import cv_func as f2 


## for download dataset
# import gdown

# print('Downloading dataset...\n')
     
# # Download the file.
# gdown.download('https://drive.google.com/uc?id=1ZYdt0zN4LjWqP3cQDblNhXjeohcryY5H', 
#                 'WomensClothingReviews.csv', 
#                 quiet=False)

# where 1 is will recommend, 0 won't

folder_path = r"C:\Users\Heng2020\OneDrive\Python NLP\NLP 05_UsefulSenLabel"
folder_Path = Path(folder_path)
df_name = 'BigBangSentenceS06_label.xlsm'


df_path = folder_Path / df_name
y_name = 'usefulness'
text_col = 'portuguese'

alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"


drop_col01 = ['episode','translation']
drop_col02 = []
drop_col03 = []
drop_col = drop_col01


data_ori = pd.read_excel(df_path)

mySeed = 20
num_to_cat_col = []

n_data = 10_000

if isinstance(n_data, str) or data_ori.shape[0] < n_data :
    data = data_ori
else:
    data = data_ori.sample(n=n_data,random_state=mySeed)


# saved_model_path = folder_path + "/" + saved_model_name + ".joblib"
data = data.drop(drop_col,axis=1)

################################### Pre processing - specific to this data(Blood Pressure) ##################
def pd_preprocess(data):
    
    df_clean = data.dropna(subset=['usefulness'])

    return df_clean



#---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------
data = pd_preprocess(data)

data = f.pd_num_to_cat(data,num_to_cat_col)
cat_col = f.pd_cat_column(data)
data = f.pd_to_category(data)

X_train, X_test, y_train, y_test = train_test_split(
                                        data[text_col], 
                                        data[y_name], 
                                        test_size=0.2, 
                                        random_state=mySeed)

vectorizer = CountVectorizer(stop_words='portuguese')

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

y_pred_train = cross_val_predict(clf, X_train_vect, y_train, cv=5)
y_pred_train = pd.DataFrame({'y_predict':y_pred_train})

accuracy = accuracy_score(y_train, y_pred_train)

scores = cross_val_score(clf, X_train_vect, y_train, cv=5,)

playsound(alarm_path)
