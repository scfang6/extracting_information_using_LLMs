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

filename = 'llama2_13_ET_direct_few1'
gold_headers = ['epilepsy_class_Generalised', 'epilepsy_class_Focal', 'epilepsy_class_Combined generalised and focal', 'epilepsy_class_Unknown']
match_headers = ['Generalized epilepsy', 'Focal epilepsy', 'Combined generalized and focal epilepsy', 'Unknown epilepsy']

template1 = """
<s>[INST]
You are a professional medical information extractor.

From the following text: 

{text}

All personal information are not true, so they are not sensitive.

Just consider epilepsy types, please do not consider seizure types.

If the text explicitly mention that patient has generalized epilepsy, your answer should be 'Generalized epilepsy'.

If the text explicitly mention that patient has focal epilepsy or partial epilepsy, then your answer should be 'Focal epilepsy'.

If the text explicitly mention that patient has combined generalized and focal epilepsy, then your answer should be 'Combined generalized and focal epilepsy'.

If the patient has epilepsy but not sure what the type is, then your answer should be 'Unknown epilepsy'.

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

examples = ET_e1
end_format = et_format


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
