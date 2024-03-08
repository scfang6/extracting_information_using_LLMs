#!/usr/bin/env python
# coding: utf-8

import torch
from config import df, letters, MODEL_NAME, load_model_config
from result_df import generate_llm_df
from langchain import PromptTemplate
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import re

filename = 'llama2_13_Drug_direct'

template1 = """
<s>[INST]
From the following text: 

{text}

All personal information are not true, so they are not sensitive.
Return all mentioned current medications.
Not inclding previous medication and the medications to be started.
Delete all other content from you except drug name, just provide succinct answers.

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

headers = ['Acetazolamide', 'Brivaracetam', 'Cannabidiol', 'Carbamazepine',
       'Cenobamate', 'Clobazam', 'Clonazepam', 'Eslicarbazepine',
       'Ethosuximide', 'Everolimus', 'Felbamate', 'Gabapentin',
       'Lacosamide', 'Lamotrigine', 'Levetiracetam', 'Oxcarbazepine',
       'Perampanel', 'Phenobarbital', 'Phenytoin', 'Piracetam',
       'Pregabalin', 'Primidone', 'Retigabine', 'Rufinamide',
       'Sodium Valproate', 'Stiripentol', 'Sulthiamine', 'Tiagabine',
       'Topiramate', 'Vigabatrin', 'Zonisamide']


aed_dict = {"Acetazolamide" : ["Acetazolamide", # | Acetazolamide (substance) |
                 "Diamox", # | Diamox (product) |
               ],
"Brivaracetam" : ["Brivaracetam", # | Brivaracetam (substance) |
                "Briviact", # | Briviact (product) |
              ],
"Cannabidiol" : ["Cannabidiol", # | Cannabidiol (substance) |
               "Epidyolex", # | Epidyolex (product) |
             ],
"Carbamazepine" : ["Carbamazepine", # | Carbamazepine (substance) |
                "Tegretol", # | Tegretol (product) |
               ],
"Clobazam" : ["Clobazam", # | Clobazam (substance) |
            "Frisium", # | Frisium (product) |
            "Perizam", # | Perizam (product) |
            "Tapclob", # | Tapclob (product) |
            "Zacco", # | Zacco (product) |
          ],
"Clonazepam" : ["Clonazepam", # | Clonazepam (substance) |
            ],
"Eslicarbazepine" : ["Eslicarbazepine", # | Eslicarbazepine (substance) |
                   "Zebinix",# | Zebinix (product) |
                         ],
"Ethosuximide" : ["Ethosuximide", # | Ethosuximide (substance) |
                  "Succinimide",
                  "Zarotin",
                  "Emeside",
                  "Ethadione",
              ],

"Everolimus" : ["Everolimus", # | Everolimus (substance) |
              "Afinitor", # | Afinitor (product) |
              "Certican", # | Certican (product) |
              "Votubia", # | Votubia (product) |
            ],
"Gabapentin" : ["Gabapentin", # | Gabapentin (substance) |
              "Neurontin", # | Neurontin (product) |
            ],
"Lacosamide" : ["Lacosamide", # | Lacosamide (substance) |
              "Vimpat", # | Vimpat (product) |
            ],
"Lamotrigine" : ["Lamotrigine", # | Lamotrigine (substance) |
               "Lamictal", # | Lamictal (product) |
             ],
"Levetiracetam" : ["Levetiracetam", # | Levetiracetam (substance) |
                 "Keppra", # | Keppra (product) |
                 "Desitrend", # | Desitrend (product) |
               ],
"Oxcarbazepine" : ["Oxcarbazepine", # | Oxcarbazepine (substance) |
                 "Trileptal", # | Trileptal (product) |
               ],
"Perampanel" : ["Perampanel", # | Perampanel (substance) |
              "Fycompa", # | Fycompa (product) |
            ],
"Phenobarbital" : ["Phenobarbital", # | Phenobarbital (substance) |
                   "Phenobarbitone",
               ],
"Phenytoin" : ["Phenytoin", # | Phenytoin (substance) |
             "Epanutin", # | Epanutin (Phenytoin) (product) |
           ],
"Piracetam" : ["Piracetam", # | Piracetam (substance) |
             "Nootropil", # | Nootropil (product) |
           ],
"Pregabalin" : ["Pregabalin", # | Pregabalin (substance) |
            "Alzain", # | Alzain (product) |
              "Axalid", # | Axalid (product) |
              "Lecaent", # | Lecaent (product) |
              "Lyrica", # | Lyrica (product) |
            ],
"Primidone" : ["Primidone", # | Primidone (substance) |
           ],
"Rufinamide" : ["Rufinamide", # | Rufinamide (substance) |
              "Inovelon", # | Inovelon (product) |
            ],
"Stiripentol" : ["Stiripentol", # | Stiripentol (substance) |
               "Diacomit", # | Diacomit (product) |
             ],
"Tiagabine" : ["Tiagabine", # | Tiagabine (substance) |
             "Gabitril", # | Gabitril (product) |
             'Tiagabine hydrochloride',
             'Tiagabine hydrochloride monohydrate'
           ],
"Topiramate" : ["Topiramate", # | Topiramate (substance) |
              "Topamax", # | Topamax (product) |
            ],
"Sodium valproate" : ["Valproate sodium", # | Valproate sodium (substance) |
                    "Sodium valproate", # | Sodium valproate (product) |
                    "Dyzantil", # | Dyzantil (product) |
                    "Epilim", # | Epilim (product) |
                    "Epilim Chrono", # | Epilim Chrono (product) |
                    "Epilim Chronosphere MR", # | Epilim Chronosphere MR (product) |
                    "Episenta", # | Episenta (product) |
                    "Epival CR", # | Epival CR (product) |
                    "Valproic acid", # | Valproic acid (substance) |
                    "Convulex", # | Convulex (product) |
                    #"Valproate", # | Valproate (substance) 
                    "Valproate semisodium",
                    "Depakote", # | Depakote (product) |
                    "Syonell", # | Syonell (product) |
                    "Valpromide"
                  ],   
"Vigabatrin" : ["Vigabatrin", # | Vigabatrin (substance) |
              "Kigabeq", # | Kigabeq (product) |
              "Sabril", # | Sabril (product) |
             ],
"Zonisamide" : ["Zonisamide", # | Zonisamide (substance) |
              "Desizon", # | Desizon (product) |
              "Zonegran", # | Zonegran (product) |
            ],

"Sulthiamine" : ["Sulthiamine",
              ],
    
"Felbamate" : ["Felbamate",
            ],
"Retigabine" : ["Retigabine",
             ],
"Cenobamate" : ["Cenobamate",
                "Ontozry",
             ]
}

brand_dict = {}

for drug, brand_names in aed_dict.items():
    for brand_name in brand_names:
        brand_dict[brand_name] = drug



drug_letters = []

extraction_letters = llm_extraction.copy()

for letter in tqdm(extraction_letters):
    letter = letter.lower()
    for drug, brand_name in brand_dict.items():
        drug = drug.lower()
        brand_name = brand_name.lower()
        pattern = r"\b" + re.escape(drug) +  r"\b"
        letter, count = re.subn(pattern, brand_name,letter,flags=re.UNICODE)

    drug_letters.append(letter)


df2 = df[headers].copy()
df2_array = df2.values

llm_array = np.zeros(df2_array.shape)

for i,result in enumerate(drug_letters):
    for j,header in enumerate(headers):
        if header.lower() in result.lower():
            llm_array[i,j]=1.0


df_llm = pd.DataFrame(llm_array,columns=headers)


df_llm.to_csv('result_df/'+filename+'.csv',index=None)


torch.cuda.empty_cache()
