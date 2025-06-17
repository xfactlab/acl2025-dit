import torch
import wandb
from transformers import Trainer
import numpy as np

class PauseTrainer(Trainer):
    def __init__(self, model_name, tokenizer, with_pause=False, pause_num = 5, dynamic_num = 1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        if with_pause:
            if 'Llama-3' in model_name:
                self.split_key = '<|start_header_id|>assistant<|end_header_id|>'
            elif 'Phi-3' in model_name:
                self.split_key = '<|assistant|>'
            elif 'phi-2' in model_name or 'phi-1_5' in model_name:
                self.split_key = 'Output:'
        self._paused = with_pause
        self.tokenizer = tokenizer
        self.pause_num = pause_num
        self.dynamic_num = dynamic_num
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self._paused:
            with torch.no_grad():
                output = model(**inputs)
                logits = output.logits[0].cpu().numpy()
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(torch.tensor(logits)).cpu().numpy()
                
                losses = -np.log(probs[np.arange(len(probs)), np.array(inputs['labels'].cpu())])
                positions_to_insert_pause = np.argsort(losses)
                
                cnt = 0
                input_ids = inputs['input_ids'].cpu().numpy().tolist()
                end_period = self.pause_num
                for idx, input_id in enumerate(input_ids):
                    # end_period = [input_id != self.tokenizer.pad_token_id].count(True) // self.pause_num
                    cnt = 0
                    for position in positions_to_insert_pause[idx]:
                        if cnt == end_period:
                            break
                        temp = input_id.copy()
                        for i in range(self.dynamic_num):
                            temp.insert(position, self.tokenizer.additional_special_tokens_ids[0])
                        if "[PAUSE]" in self.tokenizer.decode(temp, skip_special_tokens=False).split(self.split_key)[0]:
                            continue
                        else:
                            input_id = temp
                            cnt += 1
                    inputs["input_ids"][idx] = torch.tensor(input_id[cnt*self.dynamic_num:]).to(self.model.device)
                    inputs["attention_mask"][idx] = torch.ne(inputs["input_ids"][idx], self.tokenizer.pad_token_id).to(self.model.device)
                    inputs["labels"][idx] = inputs["input_ids"][idx].clone()
                    inputs["labels"][idx][inputs["labels"][idx] == self.tokenizer.pad_token_id] = -100
                    inputs["labels"][idx][inputs["labels"][idx] == self.tokenizer.additional_special_tokens_ids[0]] = -100
                
            return super().compute_loss(model, inputs, return_outputs)
                
        else:
            return super().compute_loss(model, inputs, return_outputs)