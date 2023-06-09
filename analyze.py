import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

from torch.optim import AdamW

import pandas as pd
import re
from tqdm.auto import tqdm
import math
from accelerate import Accelerator


"""
Parameters
"""
model_name = 'distilbert-base-uncased'
pronoun_pairs = {"he": "she", "she": "he", "him": "her", "her": "him",
                 "his": "her", "himself": "herself", "herself": "himself"}

masculine_pronoun = ["he", "him", "himself", "his"]
feminine_pronoun = ["she", "her", "herself", "hers"]
BATCH_SIZE = 16
device = torch.device('cuda')
mps_device = torch.device("mps")

"""
Data Preparation
"""
def split_train_test(lst_of_dict, test_size=0.2):
    """Split the list of dictionaries into train and test sets.
    """
    train, test = train_test_split(lst_of_dict, test_size=test_size, random_state=42)
    return train, test

def mask_pronoun(dataset):
    # sents = [x['sentence_text'] for x in dataset]
    tokenized_sents = [x["tokens"] for x in dataset]
    mask_index = [x["g_first_index"] for x in dataset]
    labels = [x['g'] for x in dataset]
    # sent_ids = [x['uid'] for x in dataset]

    # masked_sents = [re.sub(rf"\b({pronoun})\b", "[MASK]", sent) for sent, pronoun in zip(sents, labels)]
    masked_sents = []
    for sent, id in zip(tokenized_sents, mask_index):
        # output = sent[:id] + ["[MASK]"] + sent[id+1:]
        sent[id] = "[MASK]"
        masked_sents.append(" ".join(sent))

    return masked_sents, labels

class PronounDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        # text = data['sentence_text'].lower().strip()
        # uid = data["uid"]
        text = data["sentence_text"].strip()
        pronoun = data['g']
        corpus = data["corpus"]
        # in case the pronoun is the first token
        masked_text = re.sub(rf"\b({pronoun})\b", corpus, text) # replace 'just' the pronoun with [MASK]
        masked_text = masked_text.lower().strip()
        masked_text = re.sub(rf"\b({corpus})\b", "[MASK]", masked_text)
        encoding = self.tokenizer.encode_plus(
            masked_text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # find the index of the pronoun in input_ids, note that pronouns are always by single token
        label_idx = -100
        for i in range(len(encoding['input_ids'][0])):
            if encoding['input_ids'][0][i] == self.tokenizer.mask_token_id:
                label_idx = i
                break
        
        labels = torch.empty(encoding['input_ids'].shape, dtype=torch.long).fill_(-100)
        labels[0][label_idx] = self.tokenizer.convert_tokens_to_ids(pronoun.lower())

        return {
            'text': masked_text,
            'pronoun': pronoun,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels.clone().detach(), dtype=torch.long)
        }

def get_tokenized_dataset(fname, tokenizer=None, testall=True, test_size=0.2):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = pd.read_csv(fname, index_col=0)
    dataset = dataset.to_dict('records')

    train, test = split_train_test(dataset, test_size=test_size)
    if testall:
        test = dataset  #TODO: remember to remove this in actural training
    train_dataset = PronounDataset(train, tokenizer)
    test_dataset = PronounDataset(test, tokenizer)

    return train_dataset, test_dataset

def get_tokenized_dataset_nosplit(fname, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    dataset = pd.read_csv(fname, index_col=0)

    # train, test = split_train_test(dataset.to_dict('records'))
    # train_dataset = PronounDataset(train, tokenizer)
    # test_dataset = PronounDataset(test, tokenizer)
    dataset = PronounDataset(dataset, tokenizer)

    return dataset


def get_pronoun(tokenizer, idx):
    return tokenizer.decode(idx)




def analyze_bias(eval_dataloader, model, tokenizer, to_csv_file, device=device):
    sentences = []
    stereo_prob = []
    anti_prob = []
    for input in eval_dataloader:
        outputs = model(input_ids=input['input_ids'].to(device), attention_mask=input['attention_mask'].to(device), labels=input['labels'].to(device)).logits
        mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[1]
        # itereate through the batch
        for i in range(len(input["input_ids"])):
            sentences.append(input["text"][i])
            # print("{}th sentence".format(i))
            # print(input["text"][i])
            mask_token_logits = outputs[i, mask_token_index[i], :] # 30522 dim
            probs = torch.nn.functional.softmax(mask_token_logits)
            # Pick the [MASK] candidates with the highest logits
            # top_2_tokens = torch.topk(mask_token_logits, 2, dim=0).indices.tolist()
            # top_k_weights, top_k_tokens = torch.topk(mask_token_logits, 2, dim=0, sorted=True)
            '''
            Only consider the pronoun and its opposite
            '''
            # label_idx = input["labels"][i][input["labels"][i] > 0].item()

            # label = get_pronoun(tokenizer, label_idx)
            # print("label: ", label)

            # opposite_label = pronoun_pairs[label]
            # opposite_label_idx = tokenizer.convert_tokens_to_ids([opposite_label])
            # label_prob = probs[label_idx].item()
            # opposite_label_prob = probs[opposite_label_idx].item()

            # # print(input["text"][i])
            # # print(input["pronoun"][i])
            # # for i, token in enumerate(top_k_tokens):
            # #     token_weight = top_k_weights[i]
            # #     print(f"'>>> {tokenizer.decode([token])} | weight: '", float(token_weight))
            # print("{} has probability {}".format(label, label_prob / (label_prob + opposite_label_prob)))
            # print("{} has probability {}".format(opposite_label, opposite_label_prob / (label_prob + opposite_label_prob)))
            '''
            Consider all tokens related to the pronoun's gender, and the opposites
            '''
            label_idx = input["labels"][i][input["labels"][i] > 0].item()
            label = get_pronoun(tokenizer, label_idx)
            # gender = None
            label_prob = 0
            opposite_label_prob = 0
            if label in set(masculine_pronoun):
                # gender = "Male"
                for token in masculine_pronoun:
                    token_idx = tokenizer.convert_tokens_to_ids([token])
                    label_prob += probs[token_idx].item()
                for token in feminine_pronoun:
                    token_idx = tokenizer.convert_tokens_to_ids([token])
                    opposite_label_prob += probs[token_idx].item()
            else:
                # gender = "Female"
                for token in masculine_pronoun:
                    token_idx = tokenizer.convert_tokens_to_ids([token])
                    opposite_label_prob += probs[token_idx].item()
                for token in feminine_pronoun:
                    token_idx = tokenizer.convert_tokens_to_ids([token])
                    label_prob += probs[token_idx].item()

            # print("Stereotyped has probability {}".format(label_prob / (label_prob + opposite_label_prob)))
            # print("Anti-Steretyped has probability {}".format(opposite_label_prob / (label_prob + opposite_label_prob)))
            stereo_prob.append(label_prob / (label_prob + opposite_label_prob))
            anti_prob.append(opposite_label_prob / (label_prob + opposite_label_prob))

    df_result = pd.DataFrame({"sentences": sentences, "stereo_prob": stereo_prob, "anti-stereo_prob": anti_prob})
    df_result.to_csv(to_csv_file)

def load_finetuned(path):
    return AutoModelForMaskedLM.from_pretrained(path)
    

if __name__ == "__main__":
    '''
    gold_BUG
    '''
    train_dataset, eval_dataset = get_tokenized_dataset("data/s_gold_BUG.csv")
    # train_dataset, eval_dataset = get_tokenized_dataset("data/s_full_BUG.csv")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE)

    '''
    vanilla DistillBERT prediction
    '''
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    '''
    vanilla DistillBERT prediction
    '''
    # model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    '''
    finetuned DistillBERT prediction
    '''
    model = load_finetuned("models/attention/intervened_full_BERT_all.pt").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    
    '''
    interevene attention layers (single layer)
    ''' 
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # for i in range(5):
    #     print("Intervening {}th attention layer".format(i))
    #     model = load_finetuned("models/intervened_full_BERT_singleblock_{}.pt".format(i)).to(device)
    #     analyze_bias(eval_dataloader, model, tokenizer, "data/attention/attention_bert_prediction_full_{}.csv".format(i), device=device)
        
    

    # text = "Hello, my dog is cute [MASK]"

    # inputs = tokenizer(text, return_tensors="pt")
    # token_logits = model(**inputs).logits
    # # Find the location of [MASK] and extract its logits
    # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    # mask_token_logits = token_logits[0, mask_token_index, :]
    
    # input = next(iter(eval_dataloader))
    analyze_bias(eval_dataloader, model, tokenizer, "data/vanilla_real_bert_prediction.csv")
    # analyze_bias(eval_dataloader, model, tokenizer, "data/vanilla_bert_prediction.csv")
    # analyze_bias(eval_dataloader, model, tokenizer, "data/vanilla_bert_prediction_full.csv")
    # analyze_bias(eval_dataloader, model, tokenizer, "data/finetuned_bert_prediction_full.csv")
    # analyze_bias(eval_dataloader, model, tokenizer, "data/intervene_bert_prediction_gold.csv")