from typing import List, Dict, Union

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def dataset_split_selector(data) -> List:
    """
    This is a function for automating the process of selecting data split.
    Will be further updated.
    """
    if len(data.keys()) == 1:
        return ['train']
    else:
        return ['train', 'test']

def preprocess_dataset(examples: Union[List, Dict], pause_type="math_front_ver2", do_train=False, tokenizer=None, prompt_max_length=1024):
    if ('instruction' in examples.keys()) or ('question' in examples.keys()) and do_train == True:
        prompt_key = 'instruction' if 'instruction' in examples.keys() else 'question'
        prompt = f"Instruct:{examples[prompt_key]}\nOutput:"
        for idx, item in enumerate(examples['answer']):
            sentences = sent_tokenize(item)
            for sentence in sentences:
                ## ex) [PAUSE]100 / 2 = $<<100/2=50>>50
                if pause_type == "math_front_ver1":
                    p = re.compile(r"(?<!<<)(\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?)\s*[xX*-+/=]\s*(\d+(?:\.\d+)?)(?![^<>]*>>)")
                ## ex) 100 / 2 = [PAUSE]$<<100/2=50>>50
                elif pause_type == "math_front_ver2":
                    p = re.compile(r"[\<\$]+")
                ## ex) [PAUSE]100 / 2 = [PAUSE]$<<100/2=50>>50
                elif pause_type == "both":
                    p = re.compile(r"(?<!<<)(\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?)\s*[xX\*\-\+\/\=]\s*(\d+(?:\.\d+)?)(?![^<>]*>>)|[\<\$]+")
                elif pause_type == "all":
                    p = re.compile(r"\b(?:\d+(?:\.\d+)?|\+|\-|\*|\/|\w)\b")
                else:
                    print("only math_front_ver1, math_front_ver2, both are allowed")
                    raise ValueError
                result = p.search(sentence)
                while result:
                    start, end = result.span()
                    prompt += sentence[:start]
                    prompt += "[PAUSE]"*10 + f"{result.group()}"
                    sentence = sentence[end:]
                    result = p.search(sentence)
                prompt += sentence
    elif pause_type == "math_front" and ('instruction' in examples.keys()) or ('question' in examples.keys()) and do_train == False:
        prompt_key = 'instruction' if 'instruction' in examples.keys() else 'question'
        prompt = f"Instruct:{examples[prompt_key]}\nOutput:"
        
    else:
        print("only instruction or question is allowed")
        raise ValueError
    model_inputs = tokenizer(prompt,
                            padding=True,
                            max_length=prompt_max_length,
                            truncation=True,
                            return_tensors='pt'
                                      )
        
    return model_inputs