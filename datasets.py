"""
Script for data preprocessing and generating the stereotype and anti-stereotype data for training and testing.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer,  AutoModelForMaskedLM
import csv

def generate_datasets(fname, save=False):
    df = pd.read_csv(fname, index_col=0)
    # df["sentence_text"] = df["sentence_text"].apply(lambda x: " " + x + " ")
    # Filter to keep only the sentences with one pronoun
    df = df[df["num_of_pronouns"] == 1]
    
    # Split on stereotype: -1/0/1 for anti-stereotype, neutral and stereotype sentence
    anti_df = df[df["stereotype"] == -1].reset_index()
    neutral_df = df[df["stereotype"] == 0].reset_index()
    s_df = df[df["stereotype"] == 1].reset_index()
    
    print(df.head())
    print(len(df), len(anti_df), len(neutral_df), len(s_df))
    fname = fname[5:]
    # gold: 1113 235 331 547
    # balanced: 13951 6888 0 7063
    # full: 64299 16381 16691 31227
    
    if save:
        anti_df.to_csv("data/anti_" + fname.split("/")[-1])
        neutral_df.to_csv("data/neutral_" + fname.split("/")[-1])
        s_df.to_csv("data/s_" + fname.split("/")[-1])


def mask_pronoun(df):
    """Return a list of dictionaries of sentences with the pronoun masked and the masked true pronoun.
    """
    uids = df["uid"]
    tokens = df["tokens"]
    pronoun_idx = df["g_first_index"]
    genders = df['predicted gender']
    # print(genders)
    masked_sents = []
    true_genders = []
    result = []
    # dict = {}
    for i in range(len(tokens)):
        dict = {}
        gender = genders[i]
        sent_token = tokens[i]
        sent_token = ast.literal_eval(sent_token)
        idx = int(pronoun_idx[i])
        # print(sent_token[:idx])
        # print("AFTER")
        # print(sent_token[idx+1:])
        masked_sent_token = sent_token[:idx] + ["[MASK]"] + sent_token[idx+1:]
        s = ' '.join(masked_sent_token)
        masked_sents.append(s)
        true_genders.append(gender)
        dict["label"] = gender
        dict['text'] = s
        dict["uid"] = uids[i]
        result.append(dict)
    print(true_genders)
        
    # return masked_sents, true_pronouns
    return result

def split_train_test(lst_of_dict):
    """Split the list of dictionaries into train and test sets.
    """
    train, test = train_test_split(lst_of_dict, test_size=0.2, random_state=42)
    # dataset = {"train": train, "test": test}
    # return dataset
    return train, test


def get_tokenized_dataset(fname, tokenizer=None):
    df = pd.read_csv(fname, index_col=0)
    lst_of_dict = mask_pronoun(df)
    trainset, testset = split_train_test(lst_of_dict)
    tokenized_trainset = [tokenizer.tokenize("[CLS] %s [SEP]"%e) for e in trainset]
    # return tokenized_trainset
    # fname = fname[5:]
    # tokenized_trainset.to_csv("data/" + fname + "_tokenized_train.csv")

def get_masked_dataset(fname):
    df = pd.read_csv(fname, index_col=0)
    lst_of_dict = mask_pronoun(df)
    # trainset, testset = split_train_test(lst_of_dict)
    output = pd.DataFrame(lst_of_dict)
    fname = fname[5:]
    output.to_csv("data/" + "masked_" + fname)

'''
Adopted from 2023 Winter CS224N Default Final Project
'''
# Load the data: a list of (sentence, label)
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp):
                # print(record.keys())
                sent = record['text'].lower().strip()
                sent_id = record['uid'].lower().strip()
                # masked_idx = record["g_first_idx"]
                data.append((sent, sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence_text'].lower().strip()
                sent_id = record['uid'].lower().strip()
                label = record['predicted_gender'].lower().strip()
                # masked_idx = record["g_first_idx"]
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels) # (sentence, label, id)
    else:
        return data # (sentence, id)

class GenderTestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # self.p = args
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        # labels = [x[0] for x in data]
        sent_ids = [x[-1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        mask_token_id = self.tokenizer.mask_token_id
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        # labels = torch.LongTensor(labels)

        return token_ids, attention_mask, sents, sent_ids, mask_token_id

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids, mask_token_id = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids,
            "mask_token_id": mask_token_id
        }

        return batched_data

def model_test_eval(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_pred = []
    sents = []
    sent_ids = []
    print(dataloader)
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=False)):
        b_ids, b_mask, b_sents, b_sent_ids, b_mask_id = batch['token_ids'], batch['attention_mask'], \
                                             batch['sents'], batch['sent_ids'], batch["mask_token_id"]

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        print(b_ids.shape)

        token_logits = model(b_ids, b_mask).logits
        token_logits = token_logits.detach().cpu().numpy()
        print("token logits dim: ", token_logits.shape)
        mask_token_index = torch.where(b_ids == b_mask_id)
        print("mask token index: ", mask_token_index)
        mask_token_logits = token_logits[:, mask_token_index, :]
        # preds = np.argmax(logits, axis=1).flatten()

        # token_logits = model(**inputs).logits
# # Find the location of [MASK] and extract its logits
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # Pick the [MASK] candidates with the highest logits
# top_2_tokens = torch.topk(mask_token_logits, 2, dim=1).indices[0].tolist()

    #     y_pred.extend(preds)
    #     sents.extend(b_sents)
    #     sent_ids.extend(b_sent_ids)

    # return y_pred, sents, sent_ids
    
if __name__ == "__main__":
    # get_tokenized_dataset("data/gold_BUG.csv")
    generate_datasets("data/full_BUG.csv", save=True)
    # generate_datasets("data/full_BUG.csv")
