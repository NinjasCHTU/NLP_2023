# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 05:27:31 2023

@author: Heng2020
"""

import os
from lingtrain_aligner import preprocessor, splitter, aligner, resolver, reader, helper, vis_helper
from pathlib import Path
import lingtrain_aligner
import pandas as pd

folder = Path(r"C:/Users/Heng2020/OneDrive/Python NLP/NLP 07_Sentence Alignment")
text1_input_name = "harper_lee_ru.txt"
text2_input_name = "harper_lee_en.txt"


text1_input = folder / text1_input_name
text2_input = folder / text2_input_name

with open(text1_input, "r", encoding="utf8") as input1:
    text1 = input1.readlines()

with open(text2_input, "r", encoding="utf8") as input2:
    text2 = input2.readlines()

def output_time(t_in_sec,replay ="Time spend:"):
    
    if t_in_sec >= 60:
        print(f"{replay} {t_in_sec/60:.2f} minutes")
    else:
        print(f"{replay} {int(t_in_sec)} seconds")

def sen_alignment_df(df, lang_from = None, lang_to = None):
    # medium tested
    if lang_from is None: lang_from = df.columns[0]
    if lang_to is None: lang_to = df.columns[1]
    
    text_list_from = df.iloc[:, 0].tolist()
    text_list_to = df.iloc[:, 1].tolist()
    # assume that text from is
    result = sentence_alignment(text_list_from,text_list_to,lang_from,lang_to)
    
    return result
    

def sentence_alignment(text_from,text_to, lang_from = "pt", lang_to = "en"):
    # text_from, text_to are expected to be text or list
    # medium tested, seem to work pretty well now
    
    import os
    from lingtrain_aligner import preprocessor, splitter, aligner, resolver, reader, helper, vis_helper
    from pathlib import Path
    import lingtrain_aligner
    from playsound import playsound
    import numpy as np
    from time import time
    
    alarm_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
    
    db_name = "book.db"
    
    db_path = folder / db_name

    
    models = ["sentence_transformer_multilingual", "sentence_transformer_multilingual_labse"]
    model_name = models[0]
    
    # convert to list of text_from,text_to is not list
    
        
    ts01 = time()
    if not isinstance(text_from, list):
        text1_prepared = preprocessor.mark_paragraphs(text1)
        splitted_from = splitter.split_by_sentences_wrapper(text1_prepared, lang_from)
    else:
        splitted_from = [str(x) for x in text_from if x is not np.nan ]
        # splitted_from = splitter.split_by_sentences_wrapper(text_from, lang_from)
    
    if not isinstance(text_to, list):
        
        text2_prepared = preprocessor.mark_paragraphs(text2)
        splitted_to = splitter.split_by_sentences_wrapper(text2_prepared, lang_to)
    else:
        splitted_to = [str(x) for x in text_to if x is not np.nan ]
        # splitted_to = splitter.split_by_sentences_wrapper(text_to, lang_to)

    # temp adding title, author, h1, h2 to make it work first,.... we'll look into it when this is not avaliable later
    
    
    # if lang_from == "pt" and lang_to == "en":
    #     marker = ["(No title)%%%%%title." , 
    #                "(No author)%%%%%author.", 
    #                "(No header_)%%%%%h1.", 
    #                "(No header_)%%%%%h2."]
    #     splitted_from = marker + splitted_from
    #     splitted_to = marker + splitted_to
        
        
    # Create the database and fill it.
    if os.path.isfile(db_path):
        os.unlink(db_path)
        
    aligner.fill_db(db_path, lang_from, lang_to, splitted_from, splitted_to)
    
    # batch_ids = [0,1]
    
    aligner.align_db(db_path, \
                    model_name, \
                    batch_size=100, \
                    window=40, \
                    # batch_ids=batch_ids, \
                    save_pic=False,
                    embed_batch_size=10, \
                    normalize_embeddings=True, \
                    show_progress_bar=True
                    )
    pic_name = "alignment_vis.png"
    pic_path = folder / pic_name
    vis_helper.visualize_alignment_by_db(db_path, output_path=pic_path, lang_name_from=lang_from, lang_name_to=lang_to, batch_size=400, size=(800,800), plt_show=True)
    
    # Explore the conflicts
    
    conflicts_to_solve, rest = resolver.get_all_conflicts(db_path, min_chain_length=2, max_conflicts_len=6, batch_id=-1)
    
    resolver.get_statistics(conflicts_to_solve)
    resolver.get_statistics(rest)
    
    # resolver.show_conflict(db_path, conflicts_to_solve[8])
    
    
    steps = 10
    batch_id = -1 
    
    for i in range(steps):
        conflicts, rest = resolver.get_all_conflicts(db_path, min_chain_length=2+i, max_conflicts_len=6*(i+1), batch_id=batch_id)
        resolver.resolve_all_conflicts(db_path, conflicts, model_name, show_logs=False)
        vis_helper.visualize_alignment_by_db(db_path, output_path="img_test1.png", lang_name_from=lang_from, lang_name_to=lang_to, batch_size=400, size=(600,600), plt_show=True)
    
        if len(rest) == 0: break
    
    paragraphs = reader.get_paragraphs(db_path)[0]
    
    paragraph_from_2D = paragraphs['from']
    paragraph_to_2D = paragraphs['to']

    paragraph_from_result = [item for list_1D in paragraph_from_2D for item in list_1D]
    paragraph_to_result = [item for list_1D in paragraph_to_2D for item in list_1D]
    
    paragraph_result = pd.DataFrame({lang_from:paragraph_from_result,
                                     lang_to:paragraph_to_result
                                     })
    
    ts02 = time()
    total_time = ts02-ts01
    output_time(total_time)
    
    playsound(alarm_path)
    
    return paragraph_result


# test = sentence_alignment(text1,text2,lang_from = "ru",lang_to = "en" )

    
