import utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import string

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    losses = utils.AverageMeter()
    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids, 
            mask=mask, 
            token_type_ids=token_type_ids
        )
        
        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        
def eval_fn(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))

    fin_outputs_start = []
    fin_outputs_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []

    with torch.no_grad():
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            tweet_tokens = d['tweet_tokens']
            padding_len = d['padding_len']
            orig_sentiment = d['orig_sentiment']
            orig_selected = d['orig_selected']
            orig_tweet = d['orig_tweet']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            o1, o2 = model(
                ids=ids, 
                mask=mask, 
                token_type_ids=token_type_ids
            )
            
            fin_outputs_start.append(torch.sigmoid(o1).cpu().detach().numpy())
            fin_outputs_end.append(torch.sigmoid(o2).cpu().detach().numpy())
            fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selectd.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

    fin_outputs_start = np.vstack(fin_outputs_start)
    fin_outputs_end = np.vstack(fin_outputs_end)

    threshold = 0.2
    jaccards = []
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_output_start[j, :][:-padding_len] >= threshold
            mask_end = fin_outputs_end[j, :][:-padding_len] >= threshold
        else:
            mask_start = fin_outputs_start[j, :] >= threshold
            mask_end = fin_outputs_end[j, :] >= threshold

        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]

        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0

        for mj in range(idx_start, idx_end+1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]
        
        final_output = ''
        for ot in output_tokens:
            if ot.startswith('##'):
                final_output = final_output + ot[2:]
            elif len(ot)==1 and ot in string.punctuation:
                final_output = final_output + ot
            else:
                final_output = final_output + ' ' + ot

        final_output = final_output.strip()
        if sentiment == 'neutral' or len(original_tweet.split()) < 4:
            final_output = original_tweet
        jac = utils.jaccard(target_string.strip(), final_output.strip())
        jaccards.append(jac)
    mean_jac = np.mean(jaccards)
    return mean_jac
        
            
        

    
    


        
        