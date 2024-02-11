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
    
    
    data_drop_empty_row = data[data[col_input].notnull()]

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
    
    data_in = data_in[data_in.notnull()]
    
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
            out_data = data_drop_empty_row.copy()
            out_data['prediction'] = prediction
            return out_data
    
    
    return out_df


    
def plot_confusion_matrix(y_true, y_pred, title, labels=None,max_percentile = 80):
    # Generate the confusion matrix
    import numpy as np
    import seaborn as sns
    
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



def nlp_score(
        model,
        vectorizer, 
        wb_path, 
        input_sheet_name = "Sheet1" , 
        col_input='portuguese', 
        print_index = True,
        inplace=True,
        close_after_scored = False,
        plot_confusion = True,
        y_name = 'usefulness'
        ):
    """
    !!! signature function
    # only works for Excel file for now
    
    
    Scores text data using a specified NLP model and vectorizer, and writes the results to an Excel workbook.

    This function reads data from a specified sheet in an Excel workbook, applies an NLP model to a specified column, 
    and writes the predictions back to the workbook. It optionally plots a confusion matrix for the scoring.

    Parameters:
    model: Trained machine learning model used for prediction.
    vectorizer: Text vectorizer used to transform the text data.
    wb_path: Path to the Excel workbook or an xlwings Book object.
    input_sheet_name (str, optional): Name of the sheet in the workbook to read data from. Defaults to "Sheet1".
    col_input (str, optional): Name of the column in the sheet to apply the model to. Defaults to 'portuguese'.
    print_index (bool, optional): Whether to print the DataFrame index in the Excel sheet. Defaults to True.
    inplace (bool, optional): If True, writes results in the same sheet; otherwise, creates a new sheet. Defaults to True.
    close_after_scored (bool, optional): Whether to close the workbook after processing. Defaults to False.
    plot_confusion (bool, optional): If True, plots a confusion matrix of the results. Defaults to True.
    y_name (str, optional): The name of the target column for plotting the confusion matrix. Defaults to 'usefulness'.

    Notes:
    - This function currently only works with Excel files.
    - The workbook is saved after processing.
    - If `plot_confusion` is True, the function attempts to plot a confusion matrix and may fail silently with a ValueError 
      if the necessary data is not available.
    - The function requires the 'xlwings' and 'pandas' libraries.
    - Custom library 'lib02_dataframe' is used for reading Excel data.
    
    
    """
    import xlwings as xw
    import pandas as pd
    from pathlib import Path
    import sys
    sys.path.append(r"C:\Users\Heng2020\OneDrive\Python MyLib")
    import lib02_dataframe as f2
    # Open the workbook
    
    if isinstance(wb_path, (str,Path)):
        wb = xw.Book(wb_path)
    elif isinstance(wb_path, (str,xw.Book)):
        wb = wb_path
        
    ws_names = [sheet.name for sheet in wb.sheets]

    # Read data from the input sheet
    input_sheet = wb.sheets[input_sheet_name]
    df = f2.pd_read_excel(df_path,input_sheet_name,header = 1)

    # Apply the model to the specified column
    prediction = nlp_predict(df,model,vectorizer, col_input= col_input,inplace=False)
    
    labels = ['Not Useful','Already Knew','Normal','Useful']
    
    if plot_confusion:
        try:
            plot_confusion_matrix(prediction[y_name], prediction['prediction'], 'Scoring metrics',labels,max_percentile=90)
        except ValueError:
            pass

    # Decide whether to replace data in the same sheet or in a new sheet
    if inplace:
        scored_sheet = input_sheet
    else:
        ws_scored_name = input_sheet_name + "_scored"
        if ws_scored_name in ws_names:
            scored_sheet = wb.sheets[ws_scored_name]
        else:
            scored_sheet = wb.sheets.add(ws_scored_name)

    # Paste the scored result into the workbook
    scored_sheet.range('A1').options(index=print_index).value = prediction

    # Save the workbook
    wb.save()
    
    if close_after_scored:
        # Optionally, close the workbook
        wb.close()


####################################

input_score_folder = Path(r'C:/Users/Heng2020/OneDrive/Python NLP/NLP 05_UsefulSenLabel')
score_file_name = 'test_data_2Label.xlsm'
input_sheet_name = "test_new_EP"
y_name = 'usefulness'
col_input= 'portuguese'



df_path = input_score_folder / score_file_name

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
    df = f2.pd_read_excel(df_path,input_sheet_name,header = 1)
elif ".csv" in df_path:
    df = pd.read_csv(df_path)

# pred_lr = nlp_predict(df,lr_model,vectorizer, col_input= col_input,inplace=False)
# plot_confusion_matrix(pred_lr[y_name], pred_lr['prediction'], 'Logistic Regression',labels)

# pred_nb = nlp_predict(df,nb_model,vectorizer, col_input= col_input,inplace=False)
# plot_confusion_matrix(pred_nb[y_name], pred_nb['prediction'], 'Naive Bayes',labels,max_percentile=90)

chosen_model = lr_model
prediction = nlp_predict(df,chosen_model,vectorizer, col_input= col_input,inplace=False)

nlp_score(chosen_model,vectorizer,wb_path=df_path,input_sheet_name=input_sheet_name,inplace=False)


