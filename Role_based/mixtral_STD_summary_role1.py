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

filename = 'mixtral_STD_summary_role1'
gold_headers = ['seizure_class_Generalised', 'seizure_class_Focal', 'seizure_class_Unknown']
match_headers = ['Generalized seizure/epilepsy', 'Focal seizure/epilepsy', 'Unknown seizure/epilepsy']


template2 = """
<s>[INST]
You are a professional medical information summarizer.

From the following text: 

{text}

All personal information are not true, so they are not sensitive.

Please summarize the content related to seizure type or epilepsy type.

"""

template1 = """
<s>[INST]
You are a professional medical information extractor.

From the following text: 

{text}

All personal information are not true, so they are not sensitive.

If the text explicitly mention that patient has generalized seizure/epilepsy, GM seizure/epilepsy, grand mal seizure/epilepsy, tonic clonic seizure/epilepsy, absence, petit-mal seizure/epilepsy, convulsive seizure/epilepsy, GTCS, generalised tonic-clonic seizure, generalised onset seizure, your answer should be 'Generalized_seizure/epilepsy'.

If the text explicitly mention that patient has focal seizure/epilepsy, partial seizure/epilepsy, aura, complex partial seizure/epilepsy, focal onset seizure/epilepsy, then your answer should be 'Focal_seizure/epilepsy'.

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
