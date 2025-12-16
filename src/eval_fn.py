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

def eval_fn(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
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
        target_lp = lp_slice[i, :L, :].gather(1, target_ids[:, None]).mean().item()
        # print(data[i]["probe"])
        probe = torch.tensor(mm.tokenizer(data[i]["probe"], add_special_tokens=False)["input_ids"], device=device)
        # print(probe)
        # print(torch.tensor(mm.tokenizer(data[i]["probe"], add_special_tokens=False)["input_ids"], device=device))
        probe_len = probe.size(0)
        probe_start = find_start_pos(input_ids[i], probe)
        # print([mm.tokenizer.decode([id]) for id in input_ids[i]])
        # print(input_ids[i])
        # print(probe_start)
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
        result["log_prob"] = round(target_lp, 4)
        result["pearsonr"] = round(float(target_probe_pearsonr), 4)
        result["pred"] = mm.tokenizer.decode(pred_slice[i, :L].tolist())
        results.append(result)

    return results