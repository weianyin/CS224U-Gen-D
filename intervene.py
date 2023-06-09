"""
Switch weights from attention blocks of the anti-model (DistilBERT model fine tuned 
on anti-stereotypical dataset) to the attention blocks of the original DistilBERT model
"""
import torch
from transformers import AutoModelForMaskedLM
import pandas as pd
import argparse

from analyze import model_name

def main(anti_model_pt, method="singleblock"):
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
        print(f"Loaded vanilla model for reference: {model_name}")
        vanilla0q = vanilla_model.distilbert.transformer.layer[0].attention.q_lin.weight
        
        intervention_model = AutoModelForMaskedLM.from_pretrained(model_name)
        intervention_model = intervention_model.to(device)
        print(f"Loaded vanilla model for intervention: {model_name}")
        
        if method == "changeall":
            # Change weights in all 6 attention blocks of the vanilla model
            for i in range(6):
                intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.q_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.k_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.k_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.v_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.v_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.out_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.out_lin.weight
                assert((intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight == \
                    anti_model.distilbert.transformer.layer[i].attention.q_lin.weight).all())
                
            # Save model after all blocks have been changed
            to_model_pt = f"models/intervened_BERT_all.pt"
            vanilla_model.save_pretrained(to_model_pt)
            print(f"Saved intervened model from all blocks to {to_model_pt}")
                
        elif method == "singleblock":
            for i in range(6):
                intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.q_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.k_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.k_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.v_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.v_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.out_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.out_lin.weight
                assert((intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight == \
                    anti_model.distilbert.transformer.layer[i].attention.q_lin.weight).all())
                
                # Save intervened model
                to_model_pt = f"models/intervened_full_BERT_singleblock_{i}.pt"
                vanilla_model.save_pretrained(to_model_pt)
                print(f"Saved intervened singleblock model to {to_model_pt}")
                
                # Revert change for changing next layer
                intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight = \
                    vanilla_model.distilbert.transformer.layer[i].attention.q_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.k_lin.weight = \
                    vanilla_model.distilbert.transformer.layer[i].attention.k_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.v_lin.weight = \
                    vanilla_model.distilbert.transformer.layer[i].attention.v_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.out_lin.weight = \
                    vanilla_model.distilbert.transformer.layer[i].attention.out_lin.weight
                assert((intervention_model.distilbert.transformer.layer[i].attention.v_lin.weight == \
                    vanilla_model.distilbert.transformer.layer[i].attention.v_lin.weight).all())
                
        elif method == "accumulate":
            for i in range(6):
                intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.q_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.k_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.k_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.v_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.v_lin.weight
                intervention_model.distilbert.transformer.layer[i].attention.out_lin.weight = \
                    anti_model.distilbert.transformer.layer[i].attention.out_lin.weight
                assert((intervention_model.distilbert.transformer.layer[i].attention.q_lin.weight == \
                    anti_model.distilbert.transformer.layer[i].attention.q_lin.weight).all())
                
                # Save intervened model
                to_model_pt = f"models/intervened_full_BERT_accumulate_{i}.pt"
                vanilla_model.save_pretrained(to_model_pt)
                print(f"Saved intervened accumulative model to {to_model_pt}")
                # Keep change in the current layer to accumulate
                
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_gpu", action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.use_gpu:
        print("Use GPU")
    anti_model_pt = "models/anti_full_BERT_5.pt"
    main(anti_model_pt, method="singleblock")
    main(anti_model_pt, method="accumulate")