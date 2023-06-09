"""
Finetuning
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

from torch.optim import AdamW

import pandas as pd
import re
import argparse
from tqdm.auto import tqdm
import math
from accelerate import Accelerator

from analyze import get_tokenized_dataset

"""
Parameters
"""
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 2e-5
from analyze import BATCH_SIZE, model_name

def finetune(model, train_dataloader, eval_dataloader, device):
    # TODO: Only update attention parameters
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_TRAIN_EPOCHS):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(input_ids=batch['input_ids'].to(device), 
                            attention_mask=batch['attention_mask'].to(device), 
                            labels=batch['labels'].to(device))
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
            losses.append(accelerator.gather(loss.repeat(BATCH_SIZE)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}, Loss: {torch.mean(losses)}")

    model.save_pretrained("models/anti_BERT")
    return 0


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.use_gpu:
        print("Use GPU")
    train_dataset, eval_dataset = get_tokenized_dataset("data/anti_gold_BUG.csv", 
                                                        testall=False, test_size=args.test_size)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    finetune(model, train_dataloader, eval_dataloader, device)