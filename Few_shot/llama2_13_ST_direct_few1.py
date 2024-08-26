#!/usr/bin/env python
# coding: utf-8

import torch
from config3 import df, letters, MODEL_NAME, load_model_config
from result_df import generate_llm_df
from langchain import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from tqdm import tqdm
import pandas as pd
import time
from example_generator import *
from langchain import FewShotPromptTemplate

filename = 'llama2_13_ST_direct_few1'
gold_headers = ['seizure_class_Generalised', 'seizure_class_Focal', 'seizure_class_Unknown']
match_headers = ['Generalized seizure/epilepsy', 'Focal seizure/epilepsy', 'Unknown seizure/epilepsy']

template1 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.

If the text explicitly mention that patient has generalized seizure/epilepsy, your answer should be 'Generalized_seizure/epilepsy'.

If the text explicitly mention that patient has focal seizure/epilepsy or partial seizure/epilepsy, then your answer should be 'Focal_seizure/epilepsy'.

If the patient has seizure/epilepsy but not sure what the type is, then your answer should be 'Unknown seizure/epilepsy'.

Don't extrapolate or assume, please answer my question in the following JSON format:
[
    "Generalized seizure/epilepsy":"True" or "False",
    "Focal seizure/epilepsy":"True" or "False",
    "Unknown seizure/epilepsy":"True" or "False",
]

[/INST]
"""

llm = load_model_config(MODEL_NAME)



s_time = time.time()

examples = ST_e1
end_format = ST_format


prompt1 = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix = suffix,
    input_variables=["query"],
)

a = """
    From the following text: 
    
    """



llm_extraction = []

for i, letter in enumerate(tqdm(letters)):
    q = a+letter+end_format
    result = llm(prompt1.format(query=q))
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
