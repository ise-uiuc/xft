import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from model.configuration_llama_moe_upscaling import LlamaMoEUpscalingConfig
from model.modeling_llama_moe_upscaling_hf import LlamaMoEUpscalingForCausalLM
from model.modeling_llama_weighted_hf import LlamaWeightedForCausalLM
import torch
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_weighted_to_dense(args):
    ds_coder = LlamaForCausalLM.from_pretrained(
        args.model_moe
    )
    ds_coder_state_dict = ds_coder.state_dict()
    print(ds_coder)

    ds_coder_moe = LlamaMoEUpscalingForCausalLM.from_pretrained(
        args.model_moe
    )
    ds_coder_moe_state_dict = ds_coder_moe.state_dict()

    ds_coder_weighted_1_3B = LlamaWeightedForCausalLM.from_pretrained(
        args.model_dense
    )
    ds_coder_weighted_1_3B_state_dict = ds_coder_weighted_1_3B.state_dict()

    experts_weighted_average = {}

    lambda_ = args.shared_expert_weight
    for key in ds_coder_weighted_1_3B_state_dict.keys():
        if ".mlp.average_weights" in key:
            layer_name = key.split(".mlp.average_weights")[0]
            experts_weighted_average[layer_name] = torch.nn.functional.softmax(ds_coder_weighted_1_3B_state_dict[key]).to("cpu").tolist()
            experts_weighted_average[layer_name] = [lambda_] + [x*(1-lambda_) for x in experts_weighted_average[layer_name]]

    print(experts_weighted_average)

    experts_weight = {}

    for key in ds_coder_moe_state_dict.keys():
        if ".mlp.calculator.experts." in key:
            layer_name = key.split(".mlp.calculator.experts.")[0]
            if layer_name not in experts_weight:
                experts_weight[layer_name] = {}
            if "weight_up" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_up.")[1].split(".weight")[0])
                if "up_proj" not in experts_weight[layer_name]:
                    experts_weight[layer_name]["up_proj"] = {}
                experts_weight[layer_name]["up_proj"][expert_id] = ds_coder_moe_state_dict[key]
            elif "weight_down" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_down.")[1].split(".weight")[0])
                if "down_proj" not in experts_weight[layer_name]:
                    experts_weight[layer_name]["down_proj"] = {}
                experts_weight[layer_name]["down_proj"][expert_id] = ds_coder_moe_state_dict[key]
            elif "weight_gate" in key:
                expert_id = int(key.split(".mlp.calculator.experts.weight_gate.")[1].split(".weight")[0])
                if "gate_proj" not in experts_weight[layer_name]:
                    experts_weight[layer_name]["gate_proj"] = {}
                experts_weight[layer_name]["gate_proj"][expert_id] = ds_coder_moe_state_dict[key]

    num_experts = args.num_experts
    for key in ds_coder_state_dict.keys():
        if ".mlp." in key:
            layer_name = key.split(".mlp.")[0]
            proj_name = key.split(".mlp.")[1].split(".weight")[0]
            print(key)
            ffn_weights_dense = None
            for expert_id in experts_weight[layer_name][proj_name]:
                if ffn_weights_dense is None:
                    ffn_weights_dense = experts_weighted_average[layer_name][expert_id] * experts_weight[layer_name][proj_name][expert_id]
                else:
                    ffn_weights_dense += experts_weighted_average[layer_name][expert_id] * experts_weight[layer_name][proj_name][expert_id]
            ds_coder_state_dict[key] = ffn_weights_dense
    ds_coder.load_state_dict(ds_coder_state_dict)

    ds_coder.save_pretrained(args.save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_moe", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4")
    parser.add_argument("--model_dense", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense-lambda-75-1e-5_bs_64_epoch_1")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--shared_expert_weight", type=float, default=0.75)
    parser.add_argument("--save_path", type=str, default="ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense-lambda-75-1e-5_bs_64_epoch_1-dense")
    args = parser.parse_args()
    set_seed(42)
    convert_weighted_to_dense(args)


if __name__ == "__main__":
    main()
