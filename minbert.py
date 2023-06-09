""" Finetune minibert using a stereotype dataset and an anti-stereotype dataset, respectively.
"""
import sys
sys.path.append("224U-PROJECT")
    
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM, BertTokenizer
# transformers.logging.set_verbosity_error()

# Start code from HW1 Multi-domain Sentiment:
# https://github.com/cgpotts/cs224u/blob/main/hw_sentiment.ipynb

from torch_shallow_neural_classifier import TorchShallowNeuralClassifier


BERT_HIDDEN_SIZE = 768
OUT_SIZE = 100

class BertClassifierModule(nn.Module):
    def __init__(self, 
            weights_name="prajjwal1/bert-mini"):
        super().__init__()
        self.weights_name = weights_name
        # TODO: Want to predict the gender of masked pronoun, but how to restrict the prediction to male & female?
        self.n_classes = 2
        self.bert = AutoModel.from_pretrained(self.weights_name)
        # Finetune only
        for param in self.bert.parameters():
            param.requires_grad = True

        self.classifier_layer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              self.hidden_activation,
                                              nn.Linear(self.hidden_dim, self.n_classes))


    def forward(self, indices, mask):
        out = self.bert(indices, mask)
        return self.classifier_layer(out['last_hidden_state'][:,0])
    

def train():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

if __name__ == "__main__":
    train()