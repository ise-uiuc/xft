import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from model.configuration_llama_moe_upscaling import LlamaMoEUpscalingConfig
from model.modeling_llama_moe_upscaling_hf import LlamaMoEUpscalingForCausalLM
from model.modeling_llama_weighted_hf import LlamaWeightedForCausalLM
import torch
import random
import numpy as np
from numpy import log as ln


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_moe_to_weighted(args):
    ds_coder = LlamaWeightedForCausalLM.from_pretrained(
        args.model
    )
    ds_coder_state_dict = ds_coder.state_dict()
    print(ds_coder)

    ds_coder_moe = LlamaMoEUpscalingForCausalLM.from_pretrained(
        args.model
    )
    ds_coder_moe_state_dict = ds_coder_moe.state_dict()

    experts_weight = {}
    for key in ds_coder_moe_state_dict.keys():
        if ".mlp.calculator.experts." in key:
            layer_name = key.split(".mlp.calculator.experts.")[0]
            if layer_name not in experts_weight:
                experts_weight[layer_name] = {}
            if "weight_up" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_up.")[1].split(".weight")[0])
                if "weight_up" not in experts_weight[layer_name]:
                    experts_weight[layer_name]["weight_up"] = {}
                experts_weight[layer_name]["weight_up"][expert_id] = ds_coder_moe_state_dict[key]
            elif "weight_down" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_down.")[1].split(".weight")[0])
                if "weight_down" not in experts_weight[layer_name]:
                    experts_weight[layer_name]["weight_down"] = {}
                experts_weight[layer_name]["weight_down"][expert_id] = ds_coder_moe_state_dict[key]
            elif "weight_gate" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_gate.")[1].split(".weight")[0])
                if "weight_gate" not in experts_weight[layer_name]:
                    experts_weight[layer_name]["weight_gate"] = {}
                experts_weight[layer_name]["weight_gate"][expert_id] = ds_coder_moe_state_dict[key]

    num_experts = args.num_experts
    for key in ds_coder_state_dict.keys():
        if ".mlp." in key:
            if "average_weights" in key:
                ds_coder_state_dict[key] = torch.tensor([1 for _ in range(num_experts-1)])

    ds_coder.load_state_dict(ds_coder_state_dict)
    print(ds_coder.state_dict())
    ds_coder.save_pretrained(args.save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense")
    args = parser.parse_args()
    set_seed(42)
    convert_moe_to_weighted(args)


if __name__ == "__main__":
    main()
