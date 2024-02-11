# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 13:20:43 2023

@author: Heng2020
"""
#%%
import pandas as pd
import spacy
from playsound import playsound
# verbecc has 5 languages
# verbecc doesn't seem to have problem with these verbs(pular, odiar, atuar,nadar)
import verbecc
# mlconjug3 has 6 languages
# mlconjug3: has problem with verbs: pular, odiar, atuar,nadar
import mlconjug3
from collections import OrderedDict
import json

def filter_dict(myDict,select_key):
    # should be in my lib
    ans = {key: value for key, value in myDict.items() if key in select_key}
    return ans


def reorder_dict(input_dict, new_order):
    return OrderedDict((key, input_dict[key]) for key in new_order)

def to_list(df_sr_list):
    # can only be used in this code bc it select 1st column(not all column)
    # convert pd.Dataframe, series, list, or 1string to list
    out_list = []
    # select only 1st column
    if isinstance(df_sr_list, list):
        out_list = df_sr_list
        
    elif isinstance(df_sr_list, pd.DataFrame):
        out_list = df_sr_list.iloc[:, 0].values.tolist()
        
    elif isinstance(df_sr_list, pd.Series):
        out_list = df_sr_list.tolist()
        
    elif isinstance(df_sr_list, (int,float,complex,str)):
        out_list = [df_sr_list]
    
    else:
        print("This datatype is not suppored by this function")
        return False
    
    return out_list

class Conjugator_PT():
    # used mlconjug3
    
    # verbecc will have other structure
    
    def __init__(self,conjugator):
        self.conjugator = conjugator
        
    def present(self,verb_inf):
        # TODO add subject as parameter
        verb = self.conjugator.conjugate(verb_inf)
        subject_order = ['eu','ele','nós','eles']
        conju_info = verb.conjug_info['Indicativo']['Indicativo presente']
        conju_info_ordered = reorder_dict(conju_info,subject_order)
        
        return conju_info_ordered
    
    def conju_table_1verb(self,verb_inf):
        conju_dict = self.conjug_info(verb_inf)
        out_df = self._conju_table(conju_dict)
        return out_df
        
                
    def _conju_table(self,nested_dict):
        # only works for mlconjug3

        df = pd.DataFrame()

        exclude_mood = ["Gerúndio","Imperativo","Infinitivo","Particípio"]
        count = 0
        main_subject = ["eu","ele","nós","eles"]
        rename_map = {
            "Indicativo presente"                           :	"Present Simple",
            "Indicativo pretérito perfeito simples"         :	"Past Simple",  
            "Indicativo pretérito imperfeito"               :	"(1) Pretérito Imperfeito do Indicativo",
            "Indicativo Futuro do Presente Simples"         :	"(2) Future Simple",
            "Condicional Futuro do Pretérito Simples"       :	"(3) Futuro do Pretérito do Indicativo",
            "Conjuntivo  Subjuntivo Presente"               :	"(4) Subjuntivo",
            "Conjuntivo  Subjuntivo Pretérito Imperfeito"   :	"(5) Past Subjuntivo"
            }
        
        for mood, tenses in nested_dict.items():
            
            for tense, subjects in tenses.items():
                if tense in rename_map.keys():

                    s = filter_dict(subjects, main_subject)
                    s = pd.DataFrame([s], columns=s.keys())
                    
                    new_name = rename_map[tense]
         
                    s['tense'] = new_name

                    col_order = ['tense'] + main_subject
                    s_reorder = s[col_order]
                    
                    count += 1
        
                    # Append the series to the dataframe
                    df = pd.concat([df,s_reorder])
                
        df = df.reset_index(drop=True)
        # swap row index 1&2
        df.iloc[[1, 2]] = df.iloc[[2, 1]].to_numpy()
        return df
                    
                
    
    def conjug_info(self,verb_inf):
        return self.conjugator.conjugate(verb_inf).conjug_info
    
    
    def conju_table(self,verb_list):
        # levar past simple is wrong: the result is: lev)   !!!!!
        # same problem with pular, odiar, atuar
        
        out_df = pd.DataFrame()
        
        verb_list_in = to_list(verb_list)
        
        for curr_verb in verb_list_in:
            conju_df = self.conju_table_1verb(curr_verb)
            conju_df = conju_df.set_index('tense')
            verb_conju_list = [conju_df.values.flatten().tolist()]
            
            index = pd.MultiIndex.from_arrays([[curr_verb]], names = ['verb'])
            # use .from_product
            columns = pd.MultiIndex.from_product([new_tense_name,main_subject], names=['tense','subject'] )
            df = pd.DataFrame(verb_conju_list,index=index,columns=columns)
            out_df = pd.concat([out_df,df])
        return out_df

#%%
new_tense_name = ["Present Simple",
                  "Past Simple",
                  "(1) Pretérito Imperfeito do Indicativo",
                  "(2) Future Simple",
                  "(3) Futuro do Pretérito do Indicativo",
                  "(4) Subjuntivo",
                  "(5) Past Subjuntivo"
                  ]

main_subject = ["eu","ele","nós","eles"]


alarm_path = "H:\D_Music\Sound Effect positive-logo-opener.mp3"


# Sample dataframe
csv_path = r"C:/Users/Heng2020/OneDrive/Python NLP/NLP 02_Conjugation/Set02_Present3.csv"
PT_verb = pd.read_csv(csv_path,encoding="utf-8")

PT_verb_01 = PT_verb.copy()
PT_verb_02 = PT_verb.copy()

#%%
# slow here
conjugator1 = mlconjug3.Conjugator(language='pt')
conjugator2 = verbecc.Conjugator('pt')
playsound(alarm_path)

#%%
############################ start from this line on when changing object ######################

conju01 = Conjugator_PT(conjugator1)
test01_01 = conju01.present('falar')


subject_col = main_subject * len(new_tense_name)

test03 = conju01.conju_table_1verb('levar')

verb_list = ['beber','ser','levar','pegar','estar']
test02 = conju01.conju_table(PT_verb_01)
test02.head()
playsound(alarm_path)

#%%
test01 = conjugator2.conjugate('atuar')['moods']

content = test01['condicional']

df = pd.DataFrame(columns=['tense', 'eu', 'ele', 'nós', 'eles'])

# Populate the DataFrame with content
for tense, conjugations in content.items():
    row = [tense] + [conjugation.split(' ')[1] for conjugation in conjugations if conjugation.split(' ')[0] not in ['tu', 'vós']]
    df.loc[len(df)] = row

#%%

df = pd.DataFrame(columns=['tense', 'eu', 'ele', 'nós', 'eles'])

for mood,content in test01.items():
    df_temp = pd.DataFrame(columns=['tense', 'eu', 'ele', 'nós', 'eles'])
    
    for tense, conjugations in content.items():
        haveError = False
        try:
            row = [tense] + [conjugation.split(' ')[1] for conjugation in conjugations if conjugation.split(' ')[0] not in ['tu', 'vós']]
        except IndexError:
            pass
        except Exception as e:
            pass

        try:
            df_temp.loc[len(df_temp)] = row
        except Exception as e:
            haveError = True
            pass
    if not haveError:
        df = pd.concat([df,df_temp])

df.reset_index()  

print(df)


# %%
