# This is dumb... can also just use clm.py
import numpy as np
import torch
import time
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from factscore.lm import LM

print("Reloaded mistral")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MistralModel(LM):
    
    def __init__(self, model_name, cache_file, model_and_tokenizer=None):
        super().__init__(cache_file=cache_file)
        
        self.model_name = model_name
        if model_and_tokenizer is not None:
            self.model, self.tokenizer = model_and_tokenizer
        else:
            self.model = None

    def load_model(self):
        # print("Mistral load_model called!")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir="/gscratch/xlab/blnewman/models/transformers/",
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="/gscratch/xlab/blnewman/models/transformers/")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        inputs = []
        for prompt in prompts:
            prompt_dict = [{"role": "user", "content": prompt}]
            inputs.append(
                self.tokenizer.apply_chat_template(
                    prompt_dict, return_tensors="pt",# padding="max_length", max_length=55
                ).to(DEVICE)
            )
        
        # breakpoint()
        inputs = torch.nn.utils.rnn.pad_sequence([
            inpt.flatten().flip(dims=[0]) for inpt in inputs
        ], batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        attn_mask = (inputs != self.tokenizer.pad_token_id)
        
        generated_ids = self.model.generate(
                inputs,
                attention_mask=attn_mask,
                max_new_tokens=max_output_length,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
        ).cpu()
        generations_batch = (
            self.tokenizer.batch_decode(
                generated_ids[:, inputs.shape[1]:],
                skip_special_tokens=True
            )
        )
        
        # do some light post-processing specific to mistral
        generations_batch =  [
            new_tokens.split("\n\n")[0] for new_tokens in generations_batch
        ]
        
        return generations_batch, generated_ids
            
        # # Single prompt below
        # prompt_dict = [{"role": "user", "content": prompt}]
        # inputs = self.tokenizer.apply_chat_template(prompt_dict, return_tensors="pt").to(DEVICE)
        # generated_ids = self.model.generate(
        #     inputs,
        #     max_new_tokens=max_output_length,
        #     do_sample=True,
        #     num_return_sequences=1
        # )
        
        # new_tokens_batch = self.tokenizer.batch_decode(
        #     generated_ids[:, inputs.shape[1]:],
        #     skip_special_tokens=True
        # )
        
        # post process mistral generations:
        # generation = new_tokens_batch[0].split("\n\n")[0]
        
        # return generation, generated_ids