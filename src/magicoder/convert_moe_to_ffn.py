import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from model.configuration_llama_moe_upscaling import LlamaMoEUpscalingConfig
from model.modeling_llama_moe_upscaling_hf import LlamaMoEUpscalingForCausalLM
from model.modeling_llama_weighted_hf import LlamaWeightedForCausalLM
import torch
import random
import numpy as np
from numpy import log as ln
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_moe_to_ffn(args):
    ds_coder_moe = LlamaMoEUpscalingForCausalLM.from_pretrained(
        args.model
    )
    ds_coder_moe_state_dict = ds_coder_moe.state_dict()

    experts_weight = {}

    ffn_save_folder = args.save_path
    if not os.path.exists(ffn_save_folder):
        os.mkdir(ffn_save_folder)

    for key in ds_coder_moe_state_dict.keys():
        if ".mlp.calculator.experts." in key:
            layer_name = key.split(".mlp.calculator.experts.")[0]
            layer_idx = int(layer_name.split(".")[-1])
            if layer_idx not in experts_weight:
                experts_weight[layer_idx] = {}
            if "weight_up" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_up.")[1].split(".weight")[0])
                if "weight_up" not in experts_weight[layer_idx]:
                    experts_weight[layer_idx]["weight_up"] = {}
                experts_weight[layer_idx]["weight_up"][expert_id] = ds_coder_moe_state_dict[key]
            elif "weight_down" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_down.")[1].split(".weight")[0])
                if "weight_down" not in experts_weight[layer_idx]:
                    experts_weight[layer_idx]["weight_down"] = {}
                experts_weight[layer_idx]["weight_down"][expert_id] = ds_coder_moe_state_dict[key]
            elif "weight_gate" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_gate.")[1].split(".weight")[0])
                if "weight_gate" not in experts_weight[layer_idx]:
                    experts_weight[layer_idx]["weight_gate"] = {}
                experts_weight[layer_idx]["weight_gate"][expert_id] = ds_coder_moe_state_dict[key]

    for layer_idx in experts_weight:
        torch.save(experts_weight[layer_idx], os.path.join(ffn_save_folder, f'ffn_layer_{layer_idx}.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4")
    parser.add_argument("--save_path", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_ffn")
    args = parser.parse_args()
    set_seed(42)
    convert_moe_to_ffn(args)


if __name__ == "__main__":
    main()
