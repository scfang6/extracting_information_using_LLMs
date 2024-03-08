#!/usr/bin/env python
# coding: utf-8

import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import pandas as pd

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
sample_num = 300

def load_model_config(MODEL_NAME):
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",cache_dir="/opt/notebooks/ERUK/models"
    )

    # Set configurations
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.00001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        pad_token_id = 2,
    )
    
    return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})


df = pd.read_csv('samples.csv')
df = df.head(sample_num)
letters = df['text'].values