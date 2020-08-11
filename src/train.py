import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from model import BERTBaseUncased
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    dfx = pd.read_csv(config.TRANING_FILE, nrows=1000).dropna().reset_index(drop=True)

    df_train, df_valid = model_selection.train_test_split(
        dfx, 
        test_size=0.1, 
        random_state=42, 
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.TweetDataset(
        review=df_train.review.values, 
        sentiment=df_train.sentiment.values, 
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.TRAIN_BATCH_SIZE, 
        num_workers=4
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values, selected_text=df_valid.selected_text.values,
        sentiment=df.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, 
        num_workers=1
    )

    device = torch.device('cuda')
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params':[p for n, p  in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01}, 
        {'params':[p for n, p in param_optimizer if any(nd in n for n in no_decay)], 'weight_decay':0.01}
    ]

    num_train_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )
    
    best_accuracy=0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        print(f'Jaccard score={jaccard}')
        if jaccard > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = jaccard
    



    