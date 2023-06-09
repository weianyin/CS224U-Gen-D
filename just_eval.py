import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

from torch.optim import AdamW

import pandas as pd
import re
from tqdm.auto import tqdm
import math


device = 'mps'

"""
Parameters
"""
model_name = 'distilbert-base-uncased'

"""
Data Preparation
"""
def split_train_test(lst_of_dict):
    """Split the list of dictionaries into train and test sets.
    """
    train, test = train_test_split(lst_of_dict, test_size=0.2, random_state=42)
    return train, test

def mask_pronoun(dataset):
    # sents = [x['sentence_text'] for x in dataset]
    tokenized_sents = [x["tokens"] for x in dataset]
    mask_index = [x["g_first_index"] for x in dataset]
    labels = [x['g'] for x in dataset]
    sent_ids = [x['uid'] for x in dataset]

    # masked_sents = [re.sub(rf"\b({pronoun})\b", "[MASK]", sent) for sent, pronoun in zip(sents, labels)]
    masked_sents = []
    for sent, id in zip(tokenized_sents, mask_index):
        # output = sent[:id] + ["[MASK]"] + sent[id+1:]
        sent[id] = "[MASK]"
        masked_sents.append(" ".join(sent))

    return masked_sents, labels, sent_ids

class PronounDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        # text = data['sentence_text'].lower().strip()
        uid = data["uid"]
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
            'labels': torch.tensor(labels.clone().detach(), dtype=torch.long),
            "uid": uid
        }

def get_tokenized_dataset(fname, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = pd.read_csv(fname, index_col=0)
    dataset = dataset.to_dict('records')

    train, test = split_train_test(dataset)
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

pronoun_pairs = {"he": "she", "she": "he", "him": "her", "her": "him",
                 "his": "her", "himself": "herself", "herself": "himself"}

masculine_pronoun = ["he", "him", "himself", "his"]
feminine_pronoun = ["she", "her", "herself", "hers"]

def get_pronoun(tokenizer, idx):
    return tokenizer.decode(idx)


batch_size = 16

train_dataset, eval_dataset = get_tokenized_dataset("data/s_gold_BUG.csv")

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size)


model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# text = "Hello, my dog is cute [MASK]"

# inputs = tokenizer(text, return_tensors="pt")
# token_logits = model(**inputs).logits
# # Find the location of [MASK] and extract its logits
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]

optimizer = AdamW(model.parameters(), lr=2e-5)
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
num_train_epochs = 0
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}, Loss: {torch.mean(losses)}")

model.save_pretrained("models/anti_BERT")


# input = next(iter(eval_dataloader))
for input in eval_dataloader:
    outputs = model(input_ids=input['input_ids'].to(device), attention_mask=input['attention_mask'].to(device), labels=input['labels'].to(device)).logits
    mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[1]
    # itereate through the batch
    for i in range(len(input["input_ids"])):
        print("{}th sentence".format(i))
        print(input["text"][i])
        mask_token_logits = outputs[i, mask_token_index[i], :] # 30522 dim
        probs = torch.nn.functional.softmax(mask_token_index)
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
        # label_prob = mask_token_logits[label_idx].item()
        # opposite_label_prob = mask_token_logits[opposite_label_idx].item()

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

        print("Stereotyped has probability {}".format(label_prob / (label_prob + opposite_label_prob)))
        print("Anti-Steretyped has probability {}".format(opposite_label_prob / (label_prob + opposite_label_prob)))
