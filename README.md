# Learning to Insert [PAUSE] Tokens for Better Reasoning

This repository contains the implementation of our paper "Learning to Insert [PAUSE] Tokens for Better Reasoning" (ACL Findings 2025).

**Authors**: Eunki Kim, Sangryul Kim, James Thorne

**Paper**: [arXiv:2506.03616](https://arxiv.org/abs/2506.03616)

## Abstract

To enhance reasoning capabilities, previous works have explored incorporating special-purpose tokens into the training process. These strategies strengthen the learning mechanism of transformer-based large language models (LLMs). Building on prior research, in which inserting dummy tokens consecutively just before reasoning steps can enhance effectiveness, we introduce a novel approach termed Dynamic Inserting Tokens Training (DIT). Our method identifies positions within sequences where model confidence is lowest according to token log-likelihood. Strategically inserting [PAUSE] tokens on these positions bolsters the model's predictive capabilities for subsequent tokens. Experimental results across diverse datasets and models, from the 2.7B model to the 8B model, demonstrate that DIT consistently outperforms traditional fine-tuning and previous token insertion methods. With this simple yet effective method, we achieve accuracy gains of up to 4.7%p on GSM8K, 3.23%p on AQUA-RAT, and pass@1 improvements of up to 3.4%p on MBPP datasets.

## Setup

1. Clone the repository:
```bash
git clone git@github.com:xfactlab/acl2025-pause.git
cd acl2025-pause
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Files

The model files in `src/models/` are modified versions of the original transformers package models. To use these files:

1. **Option 1 - Convert to transformers package**:
   - Copy the model files to your transformers package installation
   - Replace the corresponding files in the transformers package
   - This is recommended for development and testing

2. **Option 2 - Load from local directory**:
   - Add the `src/models` directory to your Python path
   - Import the models directly from the local directory
   - This is recommended for production use

Example of loading models:
```python
# Option 1: After converting to transformers package
from transformers import PhiForCausalLMWithPause, LlamaForCausalLMWithPause

# Option 2: Loading from local directory
import sys
sys.path.append("path/to/src/models")
from models import PhiForCausalLMWithPause, LlamaForCausalLMWithPause
```

## Training

We provide training scripts for three datasets: AQUA, MBPP, and GSM8K. Each script supports different training settings:

- Dynamic (default): Our proposed DIT method
- Original: Standard fine-tuning
- Original Paper: Baseline method from previous work
- Random: Random token insertion
- All: All positions token insertion

### Training Scripts

1. AQUA Training:
```bash
bash scripts/train_phi2_aqua.sh
```

2. MBPP Training:
```bash
bash scripts/train_phi2_mbpp.sh
```

3. GSM8K Training:
```bash
bash scripts/train_phi2_gsm8k.sh
```

To use different settings, uncomment the corresponding flag in the script:
```bash
# --original
# --original_paper
# --random
# --all
```

## Prediction

We provide a unified prediction script that works with all datasets and models:

```bash
bash scripts/predict.sh <dataset> <checkpoint_dir> [--with_pause] [--original_paper]
```

Example:
```bash
bash scripts/predict.sh gsm8k checkpoints/phi2_gsm8k_dynamic --with_pause
```

### Supported Models

- Phi-2 (default)
- Llama-2
- Phi-3

To use a different model, modify the `model_name` parameter in the scripts.

## Evaluation

We provide an evaluation script that computes accuracy for all datasets:

```bash
python evaluate.py --dataset <dataset> --predict_file <predictions_file> [--answer_file <answer_file>]
```

Example:
```bash
python evaluate.py --dataset gsm8k --predict_file predictions/phi2_gsm8k_dynamic_predict.tsv
```

## Results

Our method achieves the following improvements:
- GSM8K: +4.7%p accuracy
- AQUA-RAT: +3.23%p accuracy
- MBPP: +3.4%p pass@1

## Citation

If you use our code or findings in your research, please cite our paper:

```bibtex
@article{kim2025learning,
  title={Learning to Insert [PAUSE] Tokens for Better Reasoning},
  author={Kim, Eunki and Kim, Sangryul and Thorne, James},
  journal={arXiv preprint arXiv:2506.03616},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 