import os
import sys
import argparse
import json
import csv
from typing import List
import torch
from datasets import load_dataset
from transformers import (
    PhiForCausalLMWithPause,
    LlamaForCausalLMWithPause,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed
)
from tqdm import tqdm
from src.generation_args import default_args

def load_model_and_tokenizer(args):
    """Load model and tokenizer based on arguments."""
    device_map = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PAUSE]"]})
    tokenizer.padding_side = "left"

    pause_embedding = [[tokenizer.additional_special_tokens_ids[0]]]

    if args.with_pause:
        if "llama" in args.model_name:
            model = LlamaForCausalLMWithPause.from_pretrained(
                args.checkpoint_dir,
                fallback_threshold=args.pause_threshold,
                pause_embedding=pause_embedding,
                device_map=device_map,
            )
        else:
            model = PhiForCausalLMWithPause.from_pretrained(
                args.checkpoint_dir,            
                fallback_threshold=args.pause_threshold,
                pause_embedding=pause_embedding,
                device_map=device_map,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            device_map=device_map,
            trust_remote_code=True if "Phi-3" in args.model_name else False,
        )

    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = 1024
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_generation_config(tokenizer):
    """Create generation configuration."""
    return GenerationConfig(
        max_length=1024,
        num_return_sequences=1,
        device_map="cuda",
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

def load_dataset_by_name(dataset_name, data_dir=None):
    """Load dataset based on name."""
    if dataset_name == "gsm8k":
        return load_dataset('gsm8k', 'main')['test']
    elif dataset_name == "aqua":
        return load_dataset("json", data_files=data_dir)["train"]
    elif dataset_name == "mbpp":
        return load_dataset("mbpp", 'full')["test"]
    elif dataset_name == "humaneval":
        return load_dataset("openai_humaneval")["test"]
    else:
        raise ValueError("Invalid dataset name")

def format_prompt(model_name, prompt):
    """Format prompt based on model type."""
    if model_name == 'microsoft/phi-2' or model_name == "microsoft/phi-1_5":
        return "Instruct: " + prompt + "\nOutput: "
    elif model_name == "microsoft/Phi-3-mini-4k-instruct":
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    elif model_name == "meta-llama/Meta-Llama-3-8B":
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>"
    else:
        raise ValueError("Invalid model name")

def get_prompt_key(dataset_name):
    """Get the prompt key for the dataset."""
    prompt_keys = {
        "aqua": "question",
        "gsm8k": "question",
        "mbpp": "text",
        "humaneval": "prompt"
    }
    return prompt_keys.get(dataset_name)

def save_predictions(predictions, args, save_dir):
    """Save predictions in appropriate format."""
    if args.dataset in ["mbpp", "humaneval"]:
        with open(save_dir + "_" + args.dataset + "_predict.jsonl", "w") as f:
            for prediction in predictions:
                f.write(json.dumps({"prediction": prediction}) + "\n")
    else:
        with open(save_dir + "_predict.tsv", "w") as f:
            writer = csv.writer(f, delimiter="\t", escapechar='\\')
            writer.writerow(["prediction"])
            for prediction in predictions:
                prediction = prediction.replace("\n", " ").replace("<|endoftext|>", "").replace("</s>", "")
                writer.writerow([prediction])

def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    generation_config = get_generation_config(tokenizer)

    # Load dataset
    dataset = load_dataset_by_name(args.dataset, args.data_dir)
    prompt_key = get_prompt_key(args.dataset)

    # Generate predictions
    predictions = []
    for i in tqdm(range(len(dataset))):
        prompt = format_prompt(args.model_name, dataset[i][prompt_key])
        
        if args.original_paper:
            prompt += "[PAUSE]" * 5

        input_ids = tokenizer(prompt, return_tensors="pt", padding=False).input_ids
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
        prediction = tokenizer.decode(output[0].cpu().numpy(), skip_special_tokens=False)
        predictions.append(prediction)

    # Save predictions
    if "checkpoint" in args.checkpoint_dir.split("/")[-1]:
        save_dir = os.path.join(args.save_dir, args.checkpoint_dir.split("/")[-2]) + ("_" + args.checkpoint_dir.split("/")[-1])
    else:
        save_dir = os.path.join(args.save_dir, args.checkpoint_dir.split("/")[-1])

    if args.with_pause:
        save_dir += ("_with_pause_" + "_".join(str(args.pause_threshold).split(".")))
    if args.original_paper:
        save_dir += "_original_paper"

    save_predictions(predictions, args, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = default_args(parser)
    set_seed(args.seed)
    main(args)