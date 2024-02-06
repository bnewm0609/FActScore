# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CLM(LM):
    def __init__(self, model_name, model_dir, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    
    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        # input_ids = self.tokenizer(prompts).input_ids
        # if verbose:
        #     input_ids = tqdm(input_ids)
        # inputs = []
        # for prompt in prompts:
        #     inputs.append(
        #         self.tokenizer(
        #             prompt, return_tensors="pt",# padding="max_length", max_length=55
        #         ).to(DEVICE).input_ids
        #     )

        # breakpoint()
        # pad on the left
        # inputs = torch.nn.utils.rnn.pad_sequence([
        #     inpt.flatten().flip(dims=[0]) for inpt in inputs
        # ], batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        # attn_mask = (inputs != self.tokenizer.pad_token_id)
        
        # This is unintuitive, but left padding and *not* including the
        # attention mask is what works for batch generation
        # right padding and including attention mask doesn't work (which makes sense)
        # right padding and not including attention def doesn't work (also makes sense)
        # left padding and including attention *should* work, but doesn't (maybe something with position embeds?)
        # left padding and not including attention works best for some reason
        # I wonder if that's true for mistral as well...
        try:
            gen_outputs = self.model.generate(
                    input_ids=inputs.input_ids, # inputs,
                    # attention_mask=attn_mask,
                    max_new_tokens=max_output_length,
                    return_dict_in_generate=True,
                    output_scores=True,
                    # do_sample=True,
                    # num_return_sequences=1,
                )# .cpu()
        except torch.cuda.OutOfMemoryError:
            breakpoint()
        # gen_outputs = self.model.generate(**inputs, max_new_tokens=max_output_length)
        gen_tokens = gen_outputs["sequences"].cpu()
        # saving the logits for the very first token
        scores = gen_outputs["scores"][0].detach().cpu().numpy()
        # gen_batch = self.tokenizer.decode(gen_tokens[0, inputs.shape[-1]:])
        
        generations_batch = (
            self.tokenizer.batch_decode(
                gen_tokens[:, inputs.input_ids.shape[1]:],
            )
        )
        
        generations = []
        for gen in generations_batch:
            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]

            generations.append(gen)
        
        torch.cuda.empty_cache()
        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]

        return generations, scores
        
        # generations = []
        # scores = []
        # for curr_input_ids in input_ids:
        #     if len(curr_input_ids) > max_sequence_length - max_output_length:
        #         curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
        #     curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
        #     gen_outputs = self.model.generate(
        #         curr_input_ids,
        #         max_length=curr_input_ids.shape[1]+max_output_length,
        #         return_dict_in_generate=True,
        #         output_scores=True
        #     )
        #     gen_tokens = gen_outputs["sequences"]
        #     # saving the logits for the very first token
        #     gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
        #     gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

        #     if end_if_newline:
        #         gen = gen.split("\n")[0].strip()
        #     elif end_if_second_newline:
        #         gen = "\n".join(gen.split("\n")[:2]).strip()

        #     if verbose and len(generations)==0:
        #         print ("Input:", prompts[0])
        #         print ("Prediction:", gen)

        #     if self.model_name.startswith("llama-sni"):
        #         gen = gen.split("</s>")[0]
                
        #     generations.append(gen)
        #     scores.append(gen_scores)

        # assert len(generations)==len(prompts)==len(scores)
        # if is_single:
        #     return generations[0], scores[0]
        
        # return generations, scores

    # Original implementation:
    # def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
    #               end_if_newline=False, end_if_second_newline=False, verbose=False):
    #     is_single = type(prompts)==str
    #     if is_single:
    #         prompts = [prompts]

    #     input_ids = self.tokenizer(prompts).input_ids
    #     if verbose:
    #         input_ids = tqdm(input_ids)

    #     generations = []
    #     scores = []
    #     for curr_input_ids in input_ids:
    #         if len(curr_input_ids) > max_sequence_length - max_output_length:
    #             curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
    #         curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
    #         gen_outputs = self.model.generate(
    #             curr_input_ids,
    #             max_length=curr_input_ids.shape[1]+max_output_length,
    #             return_dict_in_generate=True,
    #             output_scores=True
    #         )
    #         gen_tokens = gen_outputs["sequences"]
    #         # saving the logits for the very first token
    #         gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
    #         gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

    #         if end_if_newline:
    #             gen = gen.split("\n")[0].strip()
    #         elif end_if_second_newline:
    #             gen = "\n".join(gen.split("\n")[:2]).strip()

    #         if verbose and len(generations)==0:
    #             print ("Input:", prompts[0])
    #             print ("Prediction:", gen)

    #         if self.model_name.startswith("llama-sni"):
    #             gen = gen.split("</s>")[0]
                
    #         generations.append(gen)
    #         scores.append(gen_scores)

    #     assert len(generations)==len(prompts)==len(scores)
    #     if is_single:
    #         return generations[0], scores[0]
        
    #     return generations, scores

