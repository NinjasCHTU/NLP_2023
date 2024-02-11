# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 05:27:31 2023

@author: Heng2020
"""

import os
from lingtrain_aligner import preprocessor, splitter, aligner, resolver, reader, helper, vis_helper
from pathlib import Path
import lingtrain_aligner


folder = Path(r"C:/Users/Heng2020/OneDrive/Python NLP/NLP 07_Sentence Alignment")
text1_input_name = "harper_lee_ru.txt"
text2_input_name = "harper_lee_en.txt"


text1_input = folder / text1_input_name
text2_input = folder / text2_input_name

with open(text1_input, "r", encoding="utf8") as input1:
    text1 = input1.readlines()

with open(text2_input, "r", encoding="utf8") as input2:
    text2 = input2.readlines()

db_name = "book.db"

db_path = folder / db_name

lang_from = "ru"
lang_to = "en"

models = ["sentence_transformer_multilingual", "sentence_transformer_multilingual_labse"]
model_name = models[0]

text1_prepared = preprocessor.mark_paragraphs(text1)
text2_prepared = preprocessor.mark_paragraphs(text2)

splitted_from = splitter.split_by_sentences_wrapper(text1_prepared, lang_from)
splitted_to = splitter.split_by_sentences_wrapper(text2_prepared, lang_to)


# Create the database and fill it.


if os.path.isfile(db_path):
    os.unlink(db_path)
    
aligner.fill_db(db_path, lang_from, lang_to, splitted_from, splitted_to)

batch_ids = [0,1]

aligner.align_db(db_path, \
                model_name, \
                batch_size=100, \
                window=40, \
                batch_ids=batch_ids, \
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

resolver.show_conflict(db_path, conflicts_to_solve[10])


steps = 3
batch_id = -1 #выровнять все доступные батчи

for i in range(steps):
    conflicts, rest = resolver.get_all_conflicts(db_path, min_chain_length=2+i, max_conflicts_len=6*(i+1), batch_id=batch_id)
    resolver.resolve_all_conflicts(db_path, conflicts, model_name, show_logs=False)
    vis_helper.visualize_alignment_by_db(db_path, output_path="img_test1.png", lang_name_from=lang_from, lang_name_to=lang_to, batch_size=400, size=(600,600), plt_show=True)

    if len(rest) == 0: break

paragraphs = reader.get_paragraphs(db_path)[0]

paragraph_from = paragraphs['from']
paragraph_to = paragraphs['to']

