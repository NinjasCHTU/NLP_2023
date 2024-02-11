# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:44:05 2024

@author: Heng2020
"""

import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import vstack
from pathlib import Path
from sklearn.model_selection import cross_val_score
import joblib

from nltk.corpus import stopwords
from pathlib import Path

import sys
sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
import lib02_dataframe as f2

def nlp_predict(data,model,tfidf_vectorizer,col_input = 'portuguese_lemma', inplace = True):
    import pandas as pd
    # vocab03 = tfidf_vectorizer.vocabulary_

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
        
    data_tfidf = tfidf_vectorizer.transform(data_in)
    prediction = model.predict(data_tfidf)
    
    # vocab04 = tfidf_vectorizer.vocabulary_
    if isinstance(data, pd.Series):
        out_df = pd.DataFrame({'sentence':data, 'prediction':prediction})
        return out_df
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            data['prediction'] = prediction
            return data
        else:
            out_data = data.copy()
            out_data['prediction'] = prediction
            return out_data
    
    
    return out_df


    
def plot_confusion_matrix(y_true, y_pred, title, labels=None,max_percentile = 80, adjusted = False, y_accept = None):
    # Generate the confusion matrix
    # if adjusted = True use adjusted confusion matrix
    import numpy as np
    import seaborn as sns
    
    if adjusted:
        cm = confusion_matrix_adj(y_true, y_pred,y_accept, labels=labels)
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate the 80th percentile of the values in the confusion matrix
    vmax = np.percentile(cm, max_percentile)

    plt.figure(figsize=(8, 6))
    
    # Determine xticklabels and yticklabels based on whether labels are provided
    if labels is None:
        xticklabels = y_true.unique()
        yticklabels = y_true.unique()
    else:
        xticklabels = labels
        yticklabels = labels

    # Create the heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def confusion_matrix_adj(y_true, y_accept, y_pred, labels=None):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    # FIX seems like the function is still not correct 
    # please do the recon before using this function
    # unusable for now
    
    """
    Compute a confusion matrix with adjustments.

    Parameters:
    y_true: Array-like of true class labels.
    y_accept: Array-like of acceptable class labels.
    y_pred: Array-like of predicted class labels.
    labels: List of label names corresponding to the classes (optional).

    Returns:
    Confusion matrix as a 2D array.
    """
    # Adjust predictions
    adjusted_pred = []
    for true, accept, pred in zip(y_true, y_accept, y_pred):
        if pred == true or pred == accept:
            adjusted_pred.append(pred)
        # elif (accept is None) or ((accept is np.nan)):
        #     adjusted_pred.append(true)
        else:
            adjusted_pred.append(true)  # Considered as predicted 'true', actual 'true'

    # Compute confusion matrix
    return confusion_matrix(y_true, adjusted_pred, labels=labels)


####################################

input_testing_folder = Path(r'C:/Users/Heng2020/OneDrive/Python NLP/NLP 05_UsefulSenLabel')
testing_name = 'test_data_2Label.xlsm'
sheet_name = "Sheet1"
y_name = 'usefulness'


df_path = input_testing_folder / testing_name

saved_model_folder = Path(r'C:/Users/Heng2020/OneDrive/Python NLP/NLP 05_UsefulSenLabel/saved_models')

lr_model_name = "Linear_Regression_balanced.joblib"
nb_model_name = "Naive Bayes_balanced"
vectorizer_name = "TfidfVectorizer"

if ".joblib" not in lr_model_name:
    lr_model_name += ".joblib"
    
if ".joblib" not in nb_model_name:
    nb_model_name += ".joblib"
    
if ".joblib" not in vectorizer_name:
    vectorizer_name += ".joblib"

# declare variables so that it could trigger intellsense in Spyder
lr_model: LogisticRegression
nb_model: MultinomialNB
vectorizer: TfidfVectorizer

lr_model_path = saved_model_folder / lr_model_name
nb_model_path = saved_model_folder / nb_model_name
vectorizer_path = saved_model_folder / vectorizer_name

#------------------------------------------------------------

lr_model = joblib.load(lr_model_path)
nb_model = joblib.load(nb_model_path)
vectorizer = joblib.load(vectorizer_path)

# lr_model.predict_proba()

if any(str(df_path).endswith('.' + ext) for ext in ["xlsm","xlsx"]):
    df = f2.pd_read_excel(df_path,sheet_name,1)
elif ".csv" in df_path:
    df = pd.read_csv(df_path)

labels = ['Not Useful','Already Knew','Normal','Useful']


pred_lr = nlp_predict(df,lr_model,vectorizer, col_input= 'portuguese',inplace=False)
plot_confusion_matrix(pred_lr[y_name], pred_lr['prediction'], 'Logistic Regression',labels)

pred_nb = nlp_predict(df,nb_model,vectorizer, col_input= 'portuguese',inplace=False)
plot_confusion_matrix(pred_nb[y_name], pred_nb['prediction'], 'Naive Bayes',labels,max_percentile=90)


pred_lr = nlp_predict(df,lr_model,vectorizer, col_input= 'portuguese',inplace=False)
# plot_confusion_matrix(pred_lr[y_name], pred_lr['prediction'], 'Logistic Regression',labels,adjusted = True,y_accept=pred_lr[y_name])

test02 = pred_lr['prediction']
test01 = confusion_matrix_adj(pred_lr[y_name],pred_lr['usefulness2'],pred_lr['prediction'],labels)
print(test01)
