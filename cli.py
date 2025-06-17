import csv
import os
import time
import wandb
import torch
import argparse
from datetime import timedelta
from datasets import load_dataset
from typing import List, Dict, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from src.args import default_args
from src.utils import preprocess_logits_for_metrics, dataset_split_selector

import nltk
import random
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re
import numpy as np
from accelerate import PartialState, Accelerator, InitProcessGroupKwargs
from src.pause_trainer import PauseTrainer

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

class DMG(object):
    def __init__(self, args) -> None:
        self.start = time.gmtime()
        self.args = args

        # Load Tokenizer
        print(">>> 1. Loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        # For training Pause token, add special token
        if self.args.mask_type == "pause":
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[PAUSE]"]})
        

        # Load Model
        print(">>> 2. Loading Model")
        if self.args.flash_attention_2:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, 
                                                              cache_dir=self.args.cache_dir,
                                                              torch_dtype=torch.bfloat16,
                                                              attn_implementation="flash_attention_2")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, 
                                                              cache_dir=self.args.cache_dir,
                                                              torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True if "Phi-3" in self.args.model_name else False,
                                                              )
        
        
            
        # Resize Token Embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.args.enable_lora:
            peft_config = LoraConfig(
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                r=self.args.lora_rank,
                task_type="CAUSAL_LM",
            )

            self.model.enable_input_require_grads()

            self.model = get_peft_model(self.model, peft_config=peft_config)
            print("     2-1. LoRA adapter applied!")
            self.model.print_trainable_parameters()
        else:
            pass
                                                          
        # Load Dataset
        print(">>> 3. Loading Dataset")
        if self.args.data_name == "gsm8k":
            self.data = load_dataset("gsm8k", "main")
        elif self.args.data_name == "gsm8k-json":
            self.train = load_dataset("json", data_files=self.args.train_data_dir, )["train"]
            self.test = load_dataset("gsm8k", "main")["test"]
        elif self.args.data_name == "aqua":
            self.train = load_dataset("json", data_files=self.args.train_data_dir)["train"]
            self.test = load_dataset("json", data_files=self.args.test_data_dir)["train"]
        elif self.args.data_name == "mbpp":
            self.data = load_dataset("mbpp", 'full')
        elif self.args.data_name == "squad":
            self.data = load_dataset("rajpurkar/squad")
            # self.train = load_dataset("mbpp")["train"]
            # self.test = load_dataset("mbpp")["validation"]
            
        else:
            self.data = load_dataset(self.args.data_name, cache_dir=self.args.cache_dir)

        # Preprocess Dataset
        # Filter if the prompt length is longer than the max prompt length
        # Preprocess the dataset with format "Instruct: <instruction>\nOutput: <answer>"
        print(">>> 4. Filtering and Preprocessing Dataset")
        
        # Extract columns from the dataset(train, test)
        

        # If only training is selected, split the dataset into training set
        if self.args.data_name == "aqua":
            data_split = ['train', 'test']
        elif self.args.data_name == "mbpp" or self.args.data_name == "squad":
            data_split = ['train', 'validation']
        elif self.args.data_name == "gsm8k-json":
            data_split = ['train', 'test']
        else:
            data_split = dataset_split_selector(self.data)
        
        if self.args.do_train and not self.args.do_test and len(data_split) == 1:
            self.is_test = False
            train_split = data_split[0]
            print(f"   >>> Only Train = {self.is_test}")
            print(f"   >>> Test Set = {self.is_test}")
            
            train = self.data[train_split].filter(self.filter_dataset)
            print(f"\n\n>>> {len(train)} / {len(self.data[train_split])} rows left after filtering by prompt length.")
            self.train = train.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.data[train_split].column_names)      
        
        # If only testing is selected, split the dataset into testing set
        elif not self.args.do_train and self.args.do_test and len(data_split) > 1:
            self.is_test = True
            print(f"   >>> Only Test = {self.is_test}")
            test_split = data_split[1]
            test = self.data[test_split].filter(self.filter_dataset)
            self.test = test.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.data[test_split].column_names)
            
        # If both training and testing are selected, split the dataset into training and testing set
        # Many of datasets belongs here(GSM8K, SQuAD, etc.)
        else:
            self.is_test = True
            print(f"   >>> Test Set = {self.is_test}")
            print(f"   >>> Train Set = {self.args.do_train}")
            train_split = data_split[0]
            test_split = data_split[1]

            # filter and preprocess the test dataset
            if self.args.data_name == "aqua":
                test = self.test.filter(self.filter_dataset)
                train = self.train.filter(self.filter_dataset)
                self.test = test.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.test.column_names)
                self.train = train.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.train.column_names)
            elif self.args.data_name == "gsm8k-json":
                test = self.test.filter(self.filter_dataset)
                train = self.train.filter(self.filter_dataset)
                self.test = test.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.test.column_names)
                self.train = train.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.train.column_names)

            else:
                test = self.data[test_split].filter(self.filter_dataset)
                train = self.data[train_split].filter(self.filter_dataset)
                print(f"\n\n>>> {len(train)} / {len(self.data[train_split])} rows left after filtering by prompt length.")
            
                self.test = test.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.data[test_split].column_names)
                self.train = train.map(self.preprocess_dataset, batched=True, num_proc=self.args.num_proc, remove_columns=self.data[train_split].column_names)
            
            # set input_ids, attention mask, labels length same within the batch
            self.train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            self.test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Check the first sample of the train and test dataset
        print("train sample", self.tokenizer.decode(self.train[0]['labels'], skip_special_tokens=False).replace("<|endoftext|>", ""))
        print("test sample", self.tokenizer.decode(self.test[0]['labels'], skip_special_tokens=False).replace("<|endoftext|>", ""))

                         
                
        # Set WANDB & Logging Configurations
        # Set the run name with the model name, dataset name, current time and run name
        self.run_name = f"{self.args.model_name.split('/')[-1]}-{self.args.data_name.split('/')[-1]}-{self.start.tm_mday}-{self.start.tm_hour}-{self.start.tm_min}-{self.args.pause_type}-{self.args.run_name}"        
        
        print(f"\n\n>>> Run Name: {self.run_name}")
        
        # Set the save directory and log directory
        if self.args.save_dir is not None:
            self.save_dir = os.path.join(self.args.save_dir, f"{self.args.data_name.split('/')[-1]}/{self.run_name}")
            self.log_dir = os.path.join(self.args.save_dir, f"{self.args.data_name.split('/')[-1]}/{self.run_name}/logs")
        else: 
            self.save_dir = os.path.join('./checkpoints/', f"{self.args.data_name.split('/')[-1]}/{self.run_name}")
            self.log_dir = os.path.join('./checkpoints/', f"{self.args.data_name.split('/')[-1]}/{self.run_name}/logs")
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def preprocess_dataset(self, examples: Union[List, Dict]):
        
        # If the dataset has 'instruction' or 'question' key, the prompt is set as "Instruct: <instruction>\nOutput: <answer>"
        # And do training, the pause token is inserted in the front of the math expression based on selected pause type
        if ('instruction' in examples.keys()) or ('question' in examples.keys()) or ('text' in examples.keys()) or ('prompt' in examples.keys()) and self.args.do_train == True:
            if 'instruction' in examples.keys():
                prompt_key = 'instruction'
            elif 'question' in examples.keys():
                if self.args.data_name == "squad":
                    prompt_key = 'context'
                prompt_key = 'question'
            elif 'prompt' in examples.keys():
                prompt_key = 'prompt'
            elif 'text' in examples.keys():
                prompt_key = 'text'

            else:
                print("only instruction or question is allowed")
                raise ValueError

            if 'answer' in examples.keys():
                answer_key = 'answer'
            elif 'rationale' in examples.keys():
                answer_key = 'rationale'
            elif 'code' in examples.keys():
                answer_key = 'code'
            elif 'answers' in examples.keys():
                answer_key = 'answers'
            
            if self.args.model_name == "microsoft/phi-2" or self.args.model_name == "microsoft/phi-1_5":                
                prompt = [f"Instruct: {item}\nOutput: " for item in examples[prompt_key]]
            elif self.args.model_name == "microsoft/Phi-3-mini-4k-instruct":
                prompt = [f"<|user|>\n{item}\n<|assistant|>\n" for item in examples[prompt_key]]
            elif self.args.model_name == "meta-llama/Meta-Llama-3-8B":
                prompt = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{item}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>" for item in examples[prompt_key]]
                
            
            
            if self.args.pause_type != "None":
                for idx, item in enumerate(examples[answer_key]):
                    if self.args.data_name == "aqua":
                        if ".\n" not in item:
                            item = item.replace("\n", ".\n")
                    
                    if self.args.pause_type == "original_paper":
                        prompt[idx] += "[PAUSE]" * self.args.pause_num
                        prompt[idx] += item
                        continue
                    
                    if self.args.pause_type == "random":
                        pause_idx = random.sample(range(len(item)), self.args.pause_num)
                        cnt = 0
                        for i in pause_idx:
                            item = item[:i+cnt] + "[PAUSE]" + item[i+cnt:]
                            cnt += 1
                        prompt[idx] += item
                        continue
                    
                    if self.args.pause_type == "all":
                        prompt[idx] += ("[PAUSE]"*self.args.pause_num + " ").join(item.split(" "))
                        continue
                    
                    
                    # Seperate sentences. Because if using dynamic masking, need this setting.
                    sentences = sent_tokenize(item)                
                    for sentence in sentences:
                        
                        ## ex) [PAUSE]100 / 2 = $<<100/2=50>>50
                        if self.args.pause_type == "math_front_ver1":
                            p = re.compile(r"(?<!<<)(\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?)\s*[xX\*\-\+\/\=]\s*(\d+(?:\.\d+)?)(?![^<>]*>>)")
                        
                        ## ex) 100 / 2 = [PAUSE]$<<100/2=50>>50
                        elif self.args.pause_type == "math_front_ver2":
                            p = re.compile(r"(\$\<\<)|(\<\<)")
                        
                        ## ex) [PAUSE]100 / 2 = [PAUSE]$<<100/2=50>>50
                        elif self.args.pause_type == "both":
                            p = re.compile(r"(?<!<<)(\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?)\s*[xX\*\-\+\/\=]\s*(\d+(?:\.\d+)?)(?![^<>]*>>)|[\<\$]+")
                        
                        ## ex) [PAUSE]100 [PAUSE]/ [PAUSE]2 = $<<[PAUSE]100[PAUSE]/[PAUSE]2=[PAUSE]50>>[PAUSE]50
                        
                        # based on log-likelhood.
                        elif self.args.pause_type == "likelihood":
                            p = re.compile(r"\<\<[\w\-\+\*\/]+")
                            
                        elif self.args.pause_type == "behind_math":
                            p = re.compile(r"\>\>")
                            
                        elif self.args.pause_type == "behind_math_ver2":
                            p = re.compile(r"\>\>\d+")
                            
                        elif self.args.pause_type == "aqua_behind_math":
                            p = re.compile(r"(?<!####)[\s\$\<\>\(\)\+\-\*\/\÷\^\%\√\=\,\.\“\â\€\"\'\d]+([\(\<\>\)\+\-\*\/\÷\^\%\√\=\,\.\“\â\$\€\"\'\d]+)")
                            
                        elif self.args.pause_type == "sentence":
                            prompt[idx] += ("[PAUSE]"*self.args.pause_num + sentence + " ")
                            continue
                        
                        elif self.args.pause_type == "reasoning_front":
                            if ">>" in sentence:
                                prompt[idx] += ("[PAUSE]"*self.args.pause_num + sentence + " ")
                            else:
                                prompt[idx] += (sentence + " ")
                            continue
                        
                        elif self.args.pause_type == "reasoning_behind":
                            if ">>" in sentence:
                                prompt[idx] += (sentence + "[PAUSE]"*self.args.pause_num+" ")
                            else:
                                prompt[idx] += (sentence + " ")
                            continue
                                
                            
                        else:
                            print("only math_front_ver1, math_front_ver2, both and all are allowed")
                            raise ValueError
                        result = p.search(sentence)
                        
                        while result:
                            if self.args.pause_type == "behind_math" or self.args.pause_type == "behind_math_ver2":
                                start, end = result.span()
                                prompt[idx] += sentence[:end]
                                prompt[idx] += "[PAUSE]"*self.args.pause_num
                                sentence = sentence[end:]
                                result = p.search(sentence)
                            else:
                                start, end = result.span()
                                prompt[idx] += sentence[:start]
                                prompt[idx] += ("[PAUSE]"*self.args.pause_num + f"{result.group()}")
                                sentence = sentence[end:]
                                result = p.search(sentence)
                        prompt[idx] += sentence + " "
                    prompt[idx] = prompt[idx].strip()
                    
                    if self.args.delete_second_math:
                        prompt[idx] = prompt[idx].replace(r"[\$\<]+[\w\-\*\+\/\=]+[\>]+", "")
                        
            # elif self.args.data_name == "sqaud":
                
            else:
                for idx, item in enumerate(examples[answer_key]):
                    prompt[idx] += item
            
        else:
            raise ValueError
    
        model_inputs = self.tokenizer(prompt,
                                      max_length=self.args.prompt_max_length,
                                      padding=True,
                                      truncation=True,
                                      return_tensors='pt',
                                      )
        # if self.args.not_pause_learning:
            # pause_token_id = self.tokenizer.convert_tokens_to_ids("[PAUSE]")
            # attention_mask = model_inputs['attention_mask']
            # for i, token_id in enumerate(model_inputs['input_ids'].squeeze().tolist()):
            #     if token_id == pause_token_id:
            #         attention_mask[0][i] = 0
            # model_inputs['attention_mask'] = torch.tensor(attention_mask)
        model_inputs['labels'] = torch.clone(model_inputs['input_ids'])
        for i, token_id in enumerate(model_inputs['input_ids'].squeeze().tolist()):
            if token_id == self.tokenizer.pad_token_id:
                model_inputs['labels'][0][i] = -100
        
        return model_inputs

    # Filter the dataset by prompt length
    def filter_dataset(self, examples: Union[List, Dict]):
        if 'instruction' in examples.keys():
            prompt_key = 'instruction'
        elif 'question' in examples.keys():
            if self.args.data_name == "squad":
                prompt_key = 'context'
            prompt_key = 'question'
        elif 'prompt' in examples.keys():
            prompt_key = 'prompt'
        elif 'text' in examples.keys():
            prompt_key = 'text'
        
        if 'answer' in examples.keys():
            answer_key = 'answer'
        elif 'rationale' in examples.keys():
            answer_key = 'rationale'
        elif 'code' in examples.keys():
            answer_key = 'code'
        elif 'answers' in examples.keys():
            answer_key = 'answers'
        

        if self.args.model_name == "microsoft/phi-2" or self.args.model_name == "microsoft/phi-1_5":                
            prompt = ["Instruct: " + query + "\nOutput: "+ answer for query, answer in zip(examples[prompt_key], examples[answer_key])]
        elif self.args.model_name == "microsoft/Phi-3-mini-4k-instruct":
            prompt = [f"<|user|>\n{item}\n<|assistant|>\n{answer}" for item, answer in zip(examples[prompt_key], examples[answer_key])]
        elif self.args.model_name == "meta-llama/Meta-Llama-3-8B":
            prompt = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{item}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>{answer}" for item, answer in zip(examples[prompt_key], examples[answer_key])]
        else:
            print("only instruction or question is allowed")
            raise ValueError

        prompt_length = self.tokenizer(prompt, return_tensors='pt', padding="longest").input_ids.size(-1)
                   
        if prompt_length < self.args.prompt_max_length:    
            return True
        else:
            return False

    def custom_collate_fn(self, batch):
    # Unpack the batch
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Find the maximum length in this batch
        max_len = max([len(ids) for ids in input_ids])

        # Function to pad sequences with left padding
        def pad_sequence(seq, max_len):
            pad_token_id = self.tokenizer.pad_token_id
            seq = seq.tolist()
            return torch.tensor([pad_token_id] * (max_len - len(seq)) + seq, dtype=torch.long)
        
        # Apply padding to input_ids, attention_mask, and labels
        input_ids_padded = torch.stack([pad_sequence(seq, max_len) for seq in input_ids])
        attention_mask_padded = torch.stack([pad_sequence(mask, max_len) for mask in attention_mask])
        labels_padded = torch.stack([pad_sequence(lbl, max_len) for lbl in labels])
        
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels_padded
            }
    
    def prepare_trainer(self):
        ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=10800)
            )
        
        self.accelerator = Accelerator(kwargs_handlers=[ipg_handler])
        if (self.args.wandb_entity is not None or self.args.wandb_project_name is not None) and (self.accelerator.local_process_index == 0):
            wandb.init(name=self.run_name)
        arguments = TrainingArguments(
            output_dir=self.save_dir,  # The output directory
            logging_dir=self.log_dir,
            logging_steps=50,
            save_only_model = True,
            learning_rate=self.args.lr,
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=self.args.num_train_epochs,  # number of training epochs
            per_device_train_batch_size=self.args.per_device_train_batch_size,  # batch size for training
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,  # batch size for evaluation
            evaluation_strategy=self.args.evaluation_strategy if self.is_test else 'no',  
            save_strategy=self.args.evaluation_strategy,
            optim=self.args.optim,
            warmup_steps=self.args.warmup_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=True, 
            gradient_checkpointing_kwargs={'use_reentrant': False if self.args.enable_lora else True},
            load_best_model_at_end=False,
            do_train=self.args.do_train,
            do_eval=self.args.do_test,
            lr_scheduler_type=self.args.lr_scheduler_type,
            remove_unused_columns=False,
            report_to='wandb' if (self.args.wandb_entity is not None or self.args.wandb_project_name is not None) and (self.accelerator.local_process_index == 0) else None, 
            run_name=self.run_name,
            bf16=True,
            seed=self.args.seed,
        )
        
        data_collator = self.custom_collate_fn
        if self.args.is_dynamic:
            self.trainer = PauseTrainer(
            model=self.model,
            model_name=self.args.model_name,
            tokenizer=self.tokenizer,
            pause_num=self.args.pause_num,
            dynamic_num=self.args.dynamic_num,
            with_pause=True,
            args=arguments,
            train_dataset=self.train if self.args.do_train else None,
            eval_dataset=self.test if self.args.do_test else None,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
            
        else:            
            self.trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=self.train if self.args.do_train else None,
            eval_dataset=self.test if self.args.do_test else None,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
        
    def run(self):
        print(">>> 5. Preparing Trainer")
        self.prepare_trainer()
        
        # If training is selected, train the model
        if self.args.do_train:
            self.trainer.train()
            # Saving code for FSDP
            if self.trainer.is_fsdp_enabled:
                self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
                self.trainer.save_model()
        
        # If only testing is selected, test the model -> not completed
        else:
            self.predict_dir = os.path.join(self.args.predict_dir, f"{self.args.data_name.split('/')[-1]}/{self.run_name}")
            os.makedirs(self.predict_dir, exist_ok=True)
            with open(os.path.join(self.predict_dir, "predictions.tsv"),'w') as f:
                wr = csv.writer(f, delimiter="\t")
                wr.writerow(['추론'])
            distributed_state = PartialState()
            self.model.to(distributed_state.device)
            with distributed_state.split_between_processes(self.test, apply_padding=True) as prompt:
                for i in range(len(prompt)):
                    output_ids = self.model.generate(inputs=prompt[i]['input_ids'], max_length=self.args.response_max_length, num_beams=5, num_return_sequences=1, output_scores=True)            
                    outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    with open(os.path.join(self.predict_dir, "gsm8k_phi2_mathver2_predictions.tsv"), "a") as file:
                        wr = csv.writer(file, delimiter="\t")
                        wr.writerow([outputs[0]])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMG")
    args = default_args(parser)

    # Set the random seed for the entire pipeline
    set_seed(args.seed)

    # Set WANDB configurations
    if args.wandb_entity is not None :
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_project_name is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project_name
    else:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    print("================================================================================================\n")
    print(f">>> Fine-tuning {args.model_name} with DMG on {args.data_name}\n")
    print("================================================================================================")
    print("\n\n>>> Summary:")
    print(f"    - Training Epochs     : {args.num_train_epochs}")
    print(f"    - Prompt Max Length   : {args.prompt_max_length}")
    print(f"    - Response Max Length : {args.response_max_length}")

    item = DMG(args=args)
    item.run()