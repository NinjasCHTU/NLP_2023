# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:12:29 2023

@author: Heng2020
"""
# NEXT: conju_table => conju_table => more tenses conjugation function
# check all tenses name at: Duolingo Portuguese 03 Many Tenses

import pandas as pd
# import spacy
from playsound import playsound
# verbecc has 5 languages
import verbecc
# mlconjug3 has 6 languages
import mlconjug3
from collections import OrderedDict
import json

def filter_dict(myDict,select_key):
    # should be in my lib
    ans = {key: value for key, value in myDict.items() if key in select_key}
    return ans


def reorder_dict(input_dict, new_order):
    return OrderedDict((key, input_dict[key]) for key in new_order)



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
    
    def conju_table(self,verb_inf):
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
    

alarm_path = "H:\D_Music\Sound Effect positive-logo-opener.mp3"


# Sample dataframe
csv_path = r"C:/Users/Heng2020/OneDrive/Python NLP/NLP 02_Conjugation/Set02_Present3.csv"
PT_verb = pd.read_csv(csv_path,encoding="utf-8")

PT_verb_01 = PT_verb.copy()
PT_verb_02 = PT_verb.copy()

# slow here
conjugator1 = mlconjug3.Conjugator(language='pt')
# conjugator1 = verbecc.Conjugator('pt')
playsound(alarm_path)


conju01 = Conjugator_PT(conjugator1)
test01_01 = conju01.present('falar')

new_tense_name = ["Present Simple",
                  "Past Simple",
                  "(1) Pretérito Imperfeito do Indicativo",
                  "(2) Future Simple",
                  "(3) Futuro do Pretérito do Indicativo",
                  "(4) Subjuntivo",
                  "(5) Past Subjuntivo"
                  ]

main_subject = ["eu","ele","nós","eles"]

conjugator2 = verbecc.Conjugator('pt')
subject_col = main_subject * len(new_tense_name)

verb01 = 'beber'

test01_02 = conju01.conju_table(verb01)
test01_02 = test01_02.set_index('tense')
# test01_02["verb"] = verb01

index = pd.MultiIndex.from_arrays([verb01], names = ['verb'])

columns = pd.MultiIndex.from_product([new_tense_name,main_subject], names=['tense','subject'] )


verb_list = test01_02.values.flatten().tolist()



test_transpose = test01_02.transpose()



test01_03 = test01_02.melt(id_vars=['verb','tense'],value_vars = ['eu','ele','nós','eles'], var_name = 'subject', value_name = 'conjugation')
test01_04 = test01_03.pivot(index='verb',columns=['tense','subject'],values='conjugation')
test01_04 = test01_03.unstack()

test01_02 = conjugator2.conjugate('odiar')

# Specify the file path
file_path = 'nested_dict.json'

# Save the nested dictionary to a JSON file
with open(file_path, 'w',encoding='utf-8') as json_file:
    json.dump(test01_02, json_file, indent=4,ensure_ascii=False)
    

test01_03 = conju01.present('beber')
# Create new columns with the conjugated forms for different pronouns


PT_verb_01['I do'] = PT_verb_01['Portuguese'].apply(lambda x: conjugator1.conjugate(x).conjug_info['Indicativo']['presente']['eu'])
PT_verb_01['You do'] = PT_verb_01['Portuguese'].apply(lambda x: conjugator1.conjugate(x).conjug_info['Indicativo']['Presente']['tu'])
PT_verb_01['We do'] = PT_verb_01['Portuguese'].apply(lambda x: conjugator1.conjugate(x).conjug_info['Indicativo']['Presente']['nós'])
PT_verb_01['They do'] = PT_verb_01['Portuguese'].apply(lambda x: conjugator1.conjugate(x).conjug_info['Indicativo']['Presente']['eles'])

conjugator2 = verbecc.Conjugator('pt')

verb02 = conjugator1.conjugate('estar').conjug_info
verb02_02 = conjugator1.conjugate('estar').conjug_info['Indicativo']['Indicativo presente']['nós']

PT_verb_02['I do'] = PT_verb_02['Portuguese'].apply(lambda x: conjugator2.conjugate(x).moods['Indicative']['Present'][0])
PT_verb_02['You do'] = PT_verb_02['Portuguese'].apply(lambda x: conjugator2.conjugate(x).moods['Indicative']['Present'][1])
PT_verb_02['We do'] = PT_verb_02['Portuguese'].apply(lambda x: conjugator2.conjugate(x).moods['Indicative']['Present'][3])
PT_verb_02['They do'] = PT_verb_02['Portuguese'].apply(lambda x: conjugator2.conjugate(x).moods['Indicative']['Present'][5])




