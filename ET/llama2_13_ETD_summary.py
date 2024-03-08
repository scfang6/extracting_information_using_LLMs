#!/usr/bin/env python
# coding: utf-8

import torch
from config2 import df, letters, MODEL_NAME, load_model_config
from result_df import generate_llm_df
from langchain import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from tqdm import tqdm
import pandas as pd
import time

filename = 'llama2_13_ETD_summary'
gold_headers = ['epilepsy_class_Generalised', 'epilepsy_class_Focal', 'epilepsy_class_Combined generalised and focal', 'epilepsy_class_Unknown']
match_headers = ['Generalized epilepsy', 'Focal epilepsy', 'Combined generalized and focal epilepsy', 'Unknown epilepsy']


template2 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.

Please summarize the content related to epilepsy type.

"""

template1 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.

If the text explicitly mention that patient has generalized epilepsy, GM epilepsy, grand mal epilepsy, tonic clonic epilepsy, absence, petit-mal epilepsy, convulsive epilepsy, GTCS, generalised tonic-clonic epilepsy, generalised onset epilepsy, your answer should be 'Generalized epilepsy'.

If the text explicitly mention that patient has focal epilepsy, partial epilepsy, aura, complex partial epilepsy, focal onset epilepsy, then your answer should be 'Focal epilepsy'.

If the text explicitly mention that patient has combined generalized and focal epilepsy, then your answer should be 'Combined generalized and focal epilepsy'.

If the patient has seizure/epilepsy but not sure what the type is, then your answer should be 'Unknown epilepsy'.

Don't extrapolate or assume, please answer my question in the following JSON format:
[
    "Generalized epilepsy":"True" or "False",
    "Focal epilepsy":"True" or "False",
    "Combined generalized and focal epilepsy":"True" or "False",
    "Unknown epilepsy":"True" or "False",
]

[/INST]
"""

llm = load_model_config(MODEL_NAME)


s_time = time.time()


prompt1 = PromptTemplate(
    input_variables=["text"],
    template=template1,
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template=template2,
)

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)


overall_chain = SimpleSequentialChain(chains=[chain2, chain1], verbose=False)

llm_extraction = []

for i, letter in enumerate(tqdm(letters)):
    result = overall_chain.run(letter)
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
