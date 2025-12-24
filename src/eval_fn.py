import json
import torch
from pathlib import Path
import numpy as np
from datasets import load_dataset
from itertools import chain
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import torch.nn.functional as F


def pad_and_concat(tensors, pad_value=0.0):
    ndim = tensors[0].ndim
    max_len = max(t.shape[1] for t in tensors)

    padded = []
    if ndim == 3:
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                t = F.pad(t, (0,0,0,pad_len), value=pad_value)
            padded.append(t)

        return torch.cat(padded, dim=0)
    
    elif ndim == 2:
        for t in tensors:
            pad_len = max_len - t.shape[1]
            if pad_len > 0:
                t = F.pad(t, (0, pad_len), value=pad_value)
            padded.append(t)
        return torch.cat(padded, dim=0)

def find_start_pos(seq, subseq):
    T = seq.size(0)
    L = subseq.size(0)
    for i in range(T - L, -1, -1):
        if torch.equal(seq[i:i+L], subseq):
            return i
    return -1

def get_embedding(mm,start_pos, end_pos, input_ids, embeds):
    embeddings = []
    for pos in range(start_pos, end_pos): 
        assert embeds[pos]["token_id"] == input_ids[pos], f"Token ID {input_ids[pos]} mismatch in embeddings {embeds[pos]['token_id']}"
        embeddings.append(embeds[pos]["embedding"].float())
    embedding_mean = np.mean(embeddings, axis=0)
    return embedding_mean

def control_eval_fn(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    device = "cpu"
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    pad_id = mm.tokenizer.pad_token_id
    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=pad_id)
    log_probs = F.log_softmax(logits, dim=-1)

    target_ids_list = [
        torch.tensor(mm.tokenizer(d["input_text"], add_special_tokens=False)["input_ids"], device=device)
        for d in data
    ]

    results = []
    for i in range(len(log_probs)):
        result = dict(data[i])
        lp_slice = log_probs[i, 1:1+len(target_ids_list[i]), :].gather(1, target_ids_list[i][:, None])
        result["token_id_len"] = len(target_ids_list[i])
        result["sum_log_prob"] = lp_slice.cpu().sum().item()
        result["mean_log_prob"] = lp_slice.cpu().mean().item()
        results.append(result)

    return results

def cloze_eval_fn(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    device = "cpu"
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([input.to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([prediction.to(device) for prediction in predictions], pad_value=-100)
    embeddings = list(chain.from_iterable(embeds))

    log_prob = F.log_softmax(logits, dim=-1)

    target_ids_list = [
        torch.tensor(mm.tokenizer(d["target"], add_special_tokens=False)["input_ids"], device=device)
        for d in data
    ]
    target_logit_start = torch.tensor([find_start_pos(input_ids[i], target_ids_list[i]) - 1 for i in range(input_ids.size(0))], device=device)

    target_id_len = torch.tensor([len(x) for x in target_ids_list], device=device)
    max_target_len = max(target_id_len).item()
    ar = torch.arange(max_target_len, device=device)

    B, T, V = log_prob.shape
    target_logit_pos = target_logit_start[:, None] + ar[None, :]
    target_logit_pos = target_logit_pos.clamp(min=0, max=T-1)

    lp_slice = log_prob.gather(1, target_logit_pos[:, :, None].expand(B, max_target_len, V))
    pred_slice = predictions.gather(1, target_logit_pos)

    results = []
    for i in range(B):
        L = target_id_len[i]
        target_ids = target_ids_list[i]
        mean_target_lp = lp_slice[i, :L, :].gather(1, target_ids[:, None]).mean().item()
        sum_target_lp = lp_slice[i, :L, :].gather(1, target_ids[:, None]).sum().item()
        probe = torch.tensor(mm.tokenizer(data[i]["probe"], add_special_tokens=False)["input_ids"], device=device)
        probe_len = probe.size(0)
        probe_start = find_start_pos(input_ids[i], probe)
        probe_embedding = get_embedding(
            mm,
            start_pos=probe_start,
            end_pos=probe_start+probe_len,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i],
        )
        target_embedding = get_embedding(
            mm,
            start_pos=target_logit_start[i]+1,
            end_pos=target_logit_start[i]+1+L,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i],
        )
        target_probe_pearsonr = pearsonr(target_embedding, probe_embedding)[0]

        result = dict(data[i])
        result["mean_log_prob"] = round(mean_target_lp, 4)
        result["sum_log_prob"] = round(sum_target_lp, 4)
        result["pearsonr"] = round(float(target_probe_pearsonr), 4)
        result["pred"] = mm.tokenizer.decode(pred_slice[i, :L].tolist())
        results.append(result)

    return results

def verification_eval_fn(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    device = "cpu"
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([input.to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([prediction.to(device) for prediction in predictions], pad_value=-100)

    log_prob = F.log_softmax(logits, dim=-1)

    target_ids_list = [
        torch.tensor(mm.tokenizer(d["target"], add_special_tokens=False)["input_ids"], device=device)
        for d in data
    ]
    if data[0]["target"].startswith(" "):
        yes_id = torch.tensor(mm.tokenizer(" Yes", add_special_tokens=False)["input_ids"], device=device)
        no_id = torch.tensor(mm.tokenizer(" No", add_special_tokens=False)["input_ids"], device=device)
    else:
        yes_id = torch.tensor(mm.tokenizer("Yes", add_special_tokens=False)["input_ids"], device=device)
        no_id = torch.tensor(mm.tokenizer("No", add_special_tokens=False)["input_ids"], device=device)

    target_logit_start = torch.tensor([find_start_pos(input_ids[i], target_ids_list[i]) - 1 for i in range(input_ids.size(0))], device=device)

    target_id_len = torch.tensor([len(x) for x in target_ids_list], device=device)
    max_target_len = max(target_id_len).item()
    ar = torch.arange(max_target_len, device=device)

    B, T, V = log_prob.shape
    target_logit_pos = target_logit_start[:, None] + ar[None, :]
    target_logit_pos = target_logit_pos.clamp(min=0, max=T-1)

    lp_slice = log_prob.gather(1, target_logit_pos[:, :, None].expand(B, max_target_len, V))
    pred_slice = predictions.gather(1, target_logit_pos)

    results = []
    for i in range(B):
        L = target_id_len[i]
        yes_lp = lp_slice[i, :L, :].gather(1, yes_id[:, None]).mean().item()
        no_lp = lp_slice[i, :L, :].gather(1, no_id[:, None]).mean().item()
        target_probe_pearsonr = 0

        yes_result = dict(data[i])
        yes_result["target"] = "Yes"
        yes_result["log_prob"] = round(yes_lp, 4)
        yes_result["pearsonr"] = round(float(target_probe_pearsonr), 4)
        yes_result["pred"] = mm.tokenizer.decode(pred_slice[i, :L].tolist())
        no_result = dict(data[i])
        no_result["target"] = "No"
        no_result["log_prob"] = round(no_lp, 4)
        no_result["pearsonr"] = round(float(target_probe_pearsonr), 4)
        no_result["pred"] = mm.tokenizer.decode(pred_slice[i, :L].tolist())
        results.append(yes_result)
        results.append(no_result)
    return results