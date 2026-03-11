import json
import torch
from pathlib import Path
import numpy as np
from datasets import load_dataset
from itertools import chain
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import torch.nn.functional as F

def find_start_pos(seq, subseq):
    T = seq.size(0)
    L = subseq.size(0)
    for i in range(T - L, -1, -1):
        if torch.equal(seq[i:i+L], subseq):
            return i
    return -1

def get_embedding(mm, start_pos, end_pos, input_ids, embeds):
    embeddings = []
    for pos in range(start_pos, end_pos): 
        assert embeds[pos]["token_id"] == input_ids[pos], f"Token ID {input_ids[pos]} mismatch in embeddings {embeds[pos]['token_id']}"
        embeddings.append(embeds[pos]["embedding"].float())
    embedding_mean = np.mean(embeddings, axis=0)
    return embedding_mean

def cloze_eval_fn(mm, cfg, idxs, inputs, labels, logits, predictions, embeds, weights):
    device = "cpu"
    data = load_dataset(path=f"{cfg.data.test_path}", split="train")
    data = data.add_column("idx", list(range(len(data))))
    data = data.select(idxs.tolist())

    logits = logits.to(device)
    input_ids = inputs.to(device)
    predictions = predictions.to(device)

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
        sum_target_lp = lp_slice[i, :L, :].gather(1, target_ids[:, None]).sum().item()
        if data[i].get("probe") is None:
            target_probe_pearsonr = 0
        else:
            probe = torch.tensor(mm.tokenizer(data[i]["probe"], add_special_tokens=False)["input_ids"], device=device)
            probe_len = probe.size(0)
            probe_start = find_start_pos(input_ids[i], probe)
            probe_embedding = get_embedding(
                mm,
                start_pos=probe_start,
                end_pos=probe_start+probe_len,
                input_ids=input_ids[i].tolist(),
                embeds=embeds[i],)
            target_embedding = get_embedding(
                mm,
                start_pos=target_logit_start[i]+1,
                end_pos=target_logit_start[i]+1+L,
                input_ids=input_ids[i].tolist(),
                embeds=embeds[i],)
            target_probe_pearsonr = pearsonr(target_embedding, probe_embedding)[0]

        result = dict(data[i])
        result["sum_log_prob"] = round(sum_target_lp, 4)
        result["pearsonr"] = round(float(target_probe_pearsonr), 4)
        result["pred"] = mm.tokenizer.decode(pred_slice[i, :L].tolist())
        results.append(result)
        
    return results