#!/usr/bin/env python
# coding: utf-8

import torch
from config import df, letters, MODEL_NAME, load_model_config
from result_df import generate_llm_df
from langchain import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from tqdm import tqdm
import pandas as pd
import time

filename = 'llama2_7_Seizure_summary'
gold_headers = ['seizure']
match_headers = ['seizure']

template2 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.

Please summarize the content related to seizure.

"""

template1 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.

Please answer whether the patient has seizure.

Don't extrapolate or assume, please answer my question in the following JSON format:
[
    "seizure":"True" or "False",
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
