"""
Script for data preprocessing and generating the stereotype and anti-stereotype data for training and testing.
"""
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

def generate_datasets(fname, save=False):
    df = pd.read_csv(fname, index_col=0)
    # Filter to keep only the sentences with one pronoun
    df = df[df["num_of_pronouns"] == 1]
    
    # Split on stereotype: -1/0/1 for anti-stereotype, neutral and stereotype sentence
    anti_df = df[df["stereotype"] == -1]
    neutral_df = df[df["stereotype"] == 0]
    s_df = df[df["stereotype"] == 1]
    
    print(df.head())
    print(len(df), len(anti_df), len(neutral_df), len(s_df))
    # gold: 1717 420 434 863
    # balanced: 25504 12752 0 12752
    # full: 105687 29480 22867 53340
    
    if save:
        anti_df.to_csv("data/anti_" + fname)
        neutral_df.to_csv("data/neutral_" + fname)
        s_df.to_csv("data/s_" + fname)


def mask_pronoun(df):
    """Return a list of dictionaries of sentences with the pronoun masked and the masked true pronoun.
    """
    tokens = df["tokens"]
    pronoun_idx = df["g_first_index"]
    genders = df['predicted gender']
    # print(genders)
    masked_sents = []
    true_genders = []
    result = []
    dict = {}
    for i in range(len(tokens)):
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
    return tokenized_trainset


if __name__ == "__main__":
    # get_tokenized_dataset("data/gold_BUG.csv")
    generate_datasets("data/gold_BUG.csv")
    # generate_datasets("data/full_BUG.csv")