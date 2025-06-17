import json
import csv
import re
import signal
from datasets import load_dataset
import argparse

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

def find_first_alphabet(text):
    """Find the first alphabet character (A-E) in the text."""
    match = re.search(r'[A-E]', text)
    return match.group(0) if match else None

def find_first_integer(text):
    """Find the first integer in the text."""
    match = re.search(r'\D*(\d+)', text)
    return int(match.group(1)) if match else None

def get_answer_part(text):
    """Extract the answer part after #### in the text."""
    try:
        return text.split("####")[1]
    except:
        return ""

def evaluate_aqua(answer_file, predict_file):
    """Evaluate AQUA dataset predictions."""
    with open(answer_file, "r") as json_file:
        origin_data = json.load(json_file)

    with open(predict_file, newline='') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        next(reader)
        data_list = [row[0] for row in reader]

    assert len(data_list) == len(origin_data)

    cnt = 0
    for i in range(len(data_list)):
        origin_answer = find_first_alphabet(get_answer_part(origin_data[i]["rationale"]))
        predict_answer = find_first_alphabet(get_answer_part(data_list[i]))
        if origin_answer == predict_answer:
            cnt += 1

    return cnt / len(origin_data)

def evaluate_mbpp(predict_file):
    """Evaluate MBPP dataset predictions."""
    testset = load_dataset("mbpp")["test"]
    data_list = []

    if predict_file.endswith(".jsonl"):
        with open(predict_file, "r") as f:
            for idx, line in enumerate(f):
                try:
                    code = json.loads(line)["prediction"]
                    if "Llama-3" in predict_file:
                        code = code.split("<|start_header_id|>assistant<|end_header_id|>")[1]
                    elif 'Phi-3' in predict_file:
                        code = code.split("\n<|assistant|>")[1]
                    else:
                        code = code.split("Output:")[1]

                    code = process_code(code, testset[idx])
                    data_list.append(code)
                except Exception as e:
                    print(f"Error processing line {idx}: {e}")
                    continue

    elif predict_file.endswith(".tsv"):
        with open(predict_file, newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)
            for idx, row in enumerate(reader):
                try:
                    code = row[-1].split("Output:")[1]
                    code = process_code(code, testset[idx])
                    data_list.append(code)
                except Exception as e:
                    print(f"Error processing line {idx}: {e}")
                    continue

    score = 0
    for code in data_list:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            exec(code)
            score += 1
        except Exception as e:
            pass

    return score / len(data_list)

def process_code(code, test_case):
    """Process and clean up the code."""
    code = code.replace("[PAUSE]", "").replace("<|end_of_text|>", "").replace("<|endoftext|>", "").replace("</s>", "")
    code = code.replace("     ", "\n\t")

    if "```" in code:
        code = code.split("```")[1]
    if "python " in code:
        code = code.replace("python ", "")
    if '""""""' in code:
        code = code.replace('""""""', "")

    if "def " in code:
        if "def  def" in code:
            code = code.replace("def  def", "def")
        
        temp1 = "def " + code.split("def ")[1]
        temp2 = code.split("def ")[0]
        p = re.compile(r"(from\s\w+\simport\s\w+\n)|(import\s\w+\n)")
        result = p.search(temp2)
        while result:
            start, end = result.span()
            temp1 = temp2[start:end] + "\n" + temp1
            temp2 = temp2[end:]
            result = p.search(temp2)
        
        code = temp1
        code = code.strip()
        function_name = code.split("(")[0].split("def ")[1]
        code = code.replace(function_name, test_case["test_list"][0].split("(")[0].replace("assert ", ""))
        code = code + "\n" + "\n".join(test_case["test_list"])
    else:
        code = code + "\n" + "\n".join(test_case["test_list"])

    return code.strip()

def evaluate_gsm8k(predict_file):
    """Evaluate GSM8K dataset predictions."""
    origin_data = load_dataset("gsm8k", "main")["test"]
    data_list = []

    with open(predict_file, newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        try:
            data_list = [row[-1].replace("[PAUSE]", "") for row in reader]
        except Exception as e:
            print(f"Error reading file: {e}")
            return 0

    assert len(data_list) == len(origin_data)

    cnt = 0
    for i in range(len(origin_data)):
        origin_answer = find_first_integer(get_answer_part(origin_data[i]["answer"]))
        predict_answer = find_first_integer(get_answer_part(data_list[i]))
        if origin_answer == predict_answer:
            cnt += 1

    return cnt / len(origin_data)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions on different datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['aqua', 'mbpp', 'gsm8k'],
                      help='Dataset to evaluate on')
    parser.add_argument('--answer_file', type=str, help='Path to answer file (required for AQUA)')
    parser.add_argument('--predict_file', type=str, required=True,
                      help='Path to prediction file')
    
    args = parser.parse_args()

    if args.dataset == 'aqua':
        if not args.answer_file:
            raise ValueError("Answer file is required for AQUA dataset")
        accuracy = evaluate_aqua(args.answer_file, args.predict_file)
    elif args.dataset == 'mbpp':
        accuracy = evaluate_mbpp(args.predict_file)
    elif args.dataset == 'gsm8k':
        accuracy = evaluate_gsm8k(args.predict_file)

    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 