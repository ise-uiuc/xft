import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.configuration_llama_moe_upscaling import LlamaMoEUpscalingConfig
from model.modeling_llama_moe_upscaling_hf import LlamaMoEUpscalingForCausalLM
import torch
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_dense_to_moe(args):
    ds_coder = AutoModelForCausalLM.from_pretrained(args.model)
    ds_coder_state_dict = ds_coder.state_dict()
    print(ds_coder)

    ds_coder_ffn = {}
    for key in ds_coder_state_dict.keys():
        if ".mlp." in key:
            if key.split(".mlp.")[0] not in ds_coder_ffn:
                ds_coder_ffn[key.split(".mlp.")[0]] = {}
            ds_coder_ffn[key.split(".mlp.")[0]][key.split(".mlp.")[1].split(".weight")[0]] = ds_coder_state_dict[key]
    
    ds_coder_moe_config = LlamaMoEUpscalingConfig(num_experts=8, num_selects=6, gate_type="TopKUniversalBalancedNoisyGate")
    ds_coder_moe = LlamaMoEUpscalingForCausalLM.from_pretrained(
        args.model,
        config=ds_coder_moe_config,
    )
    ds_coder_moe_state_dict = ds_coder_moe.state_dict()
    print(ds_coder_moe)

    for key in ds_coder_moe_state_dict.keys():
        if ".mlp.calculator.experts." in key:
            layer_name = key.split(".mlp.calculator.experts.")[0]
            if "weight_up" in key:
                ds_coder_moe_state_dict[key] = ds_coder_ffn[layer_name]["up_proj"]
            elif "weight_down" in key:
                ds_coder_moe_state_dict[key] = ds_coder_ffn[layer_name]["down_proj"]
            elif "weight_gate" in key:
                ds_coder_moe_state_dict[key] = ds_coder_ffn[layer_name]["gate_proj"]

    ds_coder_moe.load_state_dict(ds_coder_moe_state_dict)
    print(ds_coder_moe.state_dict())
    ds_coder_moe.save_pretrained(args.save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--save_path", type=str, default="deepseek-coder-8x1.3b-top-6-moe-base")
    args = parser.parse_args()
    set_seed(42)
    convert_dense_to_moe(args)


if __name__ == "__main__":
    main()
