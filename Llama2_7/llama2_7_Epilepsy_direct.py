#!/usr/bin/env python
# coding: utf-8

import torch
from config import df, letters, MODEL_NAME, load_model_config
from result_df import generate_llm_df
from langchain import PromptTemplate
from tqdm import tqdm
import pandas as pd
import time

filename = 'llama2_7_Epilepsy_direct'
gold_headers = ['epilepsy']
match_headers = ['epilepsy']

template1 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.

Please answer whether the patient has epilepsy.

Don't extrapolate or assume, please answer my question in the following JSON format:
[
    "epilepsy":"True" or "False",
]

[/INST]
"""

llm = load_model_config(MODEL_NAME)


s_time = time.time()


prompt1 = PromptTemplate(
    input_variables=["text"],
    template=template1,
)

llm_extraction = []

for i, letter in enumerate(tqdm(letters)):
    result = llm(prompt1.format(text=letter))
    llm_extraction.append(result)

e_time = time.time()


file =  open('time.txt','a')
file.write(filename+'\n')
file.write(str(e_time-s_time)+'\n')
file.close()


updated_letters = llm_extraction.copy()

txtresult = 'result_txt/'+filename+'.txt'
txtfile = open(txtresult, 'w')
for item in updated_letters:
    txtfile.write(item)
    txtfile.write('\n')
    txtfile.write("________\n")
txtfile.close()


df_llm = generate_llm_df(gold_headers, match_headers, updated_letters)

df_llm.to_csv('result_df/'+filename+'.csv',index=None)


torch.cuda.empty_cache()
