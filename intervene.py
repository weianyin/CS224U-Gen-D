"""
Switch weights from attention blocks of the anti-model (DistilBERT model fine tuned 
on anti-stereotypical dataset) to the attention blocks of the original DistilBERT model
"""
import torch
from transformers import AutoModelForMaskedLM
import pandas as pd
import argparse

from analyze import model_name

def main(anti_model_pt):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        # saved = torch.load(anti_model_pt)
        # config = saved['model_config']
        # anti_model = AutoModelForMaskedLM(config)
        # anti_model.load_state_dict(saved['model'])

        anti_model = AutoModelForMaskedLM.from_pretrained(anti_model_pt)
        anti_model = anti_model.to(device)
        print(f"Loaded anti-model from {anti_model_pt}")
        print("Attention block layers:")
        print(anti_model.distilbert.transformer.layer[0].attention)
        # for i in anti_model.distilbert.transformer.layer[0].attention:
        #     print(i)
        anti0q = anti_model.distilbert.transformer.layer[0].attention.q_lin.weight
        
        vanilla_model = AutoModelForMaskedLM.from_pretrained(model_name)
        vanilla_model = vanilla_model.to(device)
        print(f"Loaded vanilla model: {model_name}")
        vanilla0q = vanilla_model.distilbert.transformer.layer[0].attention.q_lin.weight
        
        # Change weights in all 6 attention blocks of the vanilla model
        for i in range(6):
            vanilla_model.distilbert.transformer.layer[i].attention.q_lin.weight = \
                anti_model.distilbert.transformer.layer[i].attention.q_lin.weight
            vanilla_model.distilbert.transformer.layer[i].attention.k_lin.weight = \
                anti_model.distilbert.transformer.layer[i].attention.k_lin.weight
            vanilla_model.distilbert.transformer.layer[i].attention.v_lin.weight = \
                anti_model.distilbert.transformer.layer[i].attention.v_lin.weight
            vanilla_model.distilbert.transformer.layer[i].attention.out_lin.weight = \
                anti_model.distilbert.transformer.layer[i].attention.out_lin.weight
            assert((vanilla_model.distilbert.transformer.layer[i].attention.q_lin.weight == \
                anti_model.distilbert.transformer.layer[i].attention.q_lin.weight).all())
            # if (i == 0):
            #     print("Mix-model weights:")
            #     print(vanilla_model.distilbert.transformer.layer[0].attention.q_lin.weight)
        to_model_pt = "models/intervened_BERT_3.pt"
        vanilla_model.save_pretrained(to_model_pt)
        print(f"Saved intervened model to {to_model_pt}")

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_gpu", action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.use_gpu:
        print("Use GPU")
    anti_model_pt = "models/anti_BERT_3.pt"
    main(anti_model_pt)