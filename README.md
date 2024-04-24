# XFT: Unlocking the Power of Code Instruction Tuning by Simply Merging Upcycled Mixture-of-Experts

<p align="left">
    <a href="https://arxiv.org/abs/2404.15247"><img src="https://img.shields.io/badge/arXiv-2404.15247-b31b1b.svg?style=for-the-badge">
</p>

> [!IMPORTANT]
> We are constantly working on cleaning the code, improving the documentation, and adding more implementation details. Plese stay tuned!

We build XFT based on the implementation of Magicoder. Please follow the original instruction of Magicoder (https://github.com/ise-uiuc/magicoder) to install your environment. To obtain XFT_DS, you need to run the code step by step as follows:

Step 1: Upcycle an MoE model from DeepSeek-Coder-1.3B Base.
```bash
export PYTHONPATH=:[YOUR_HOME_PATH]/xft/src:[YOUR_HOME_PATH]/xft/src/magicoder
cd [YOUR_HOME_PATH]/xft/src/magicoder
python convert_dense_to_moe.py \
 --model deepseek-ai/deepseek-coder-1.3b-base \
 --save_path "deepseek-coder-8x1.3b-top-6-moe-base"
```

Step 2: 
Download [Evol-Instruct dataset](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/blob/main/data-evol_instruct-decontaminated.jsonl) and put it under `xft/data` folder. 

Instruction tune the upcycled MoE model on evol-instruct dataset.
```bash
bash train_moe.sh
```

Evaluate the instruction-tuned MoE model on HumanEval(+).
```bash
bash test_moe.sh
```


Step 3: Extract FFN weights from the instruction-tuned MoE model.
```bash
python convert_moe_to_ffn.py \
 --model "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4" \
 --save_path "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_ffn"
```

Step 4: Set the `shared_expert_weight` ($lambda$) and `ffn_folder_path` (path to the folder of FFN weights) in the config file of the instruction-tuned MoE model (`ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4/config.json`) before learning the mixing coefficients.


Step 5: Initialize the mixing coefficients which aims to merge the experts in the instruction-tuned MoE model.
```bash
python convert_moe_to_weighted.py \
 --model "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4" \
 --save_path "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense" \
 --num_experts 8
```

Step 6: Learn the mixing coefficients on evol-instruct dataset.
```bash
bash train_weighted.sh
```

Step 7: Merge the instruction-tuned MoE model based on the learned mixing coefficients. Now you will get a instruction-tuned model that has the same architecture as DeepSeek-Coder-1.3B Base.
```bash
python convert_weighted_to_dense.py \
 --model_moe "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4" \
 --model_dense "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense-lambda-75-1e-5_bs_64_epoch_1" \
 --save_path "ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense-lambda-75-1e-5_bs_64_epoch_1-dense" \
 --num_experts 8 \
 --shared_expert_weight 0.75
```

Evaluate the final model on HumanEval(+).
```bash
bash test.sh
```
