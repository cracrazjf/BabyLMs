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
    for i in range(T - L + 1):
        if torch.equal(seq[i:i+L], subseq):
            return i
    return -1

def get_embedding(start_pos, end_pos, input_ids, embeds):
    embeddings = []
    for pos in range(start_pos, end_pos): 
        assert embeds[pos]["token_id"] == input_ids[pos], f"Token ID mismatch in embeddings"
        embeddings.append(embeds[pos]["embedding"].float())
    embedding_mean = np.mean(embeddings, axis=0)
    return embedding_mean

def cat_eval_A(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([input.to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([prediction.to(device) for prediction in predictions], pad_value=-100)
    embeddings = list(chain.from_iterable(embeds))

    log_prob = F.log_softmax(logits, dim=-1)

    cat_ids_list = [
        torch.tensor(mm.tokenizer(" " + d["category"], add_special_tokens=False)["input_ids"], device=device)
        for d in data
    ]
    cat_lens = torch.tensor([len(x) for x in cat_ids_list], device=device)
    cat_logit_start = torch.tensor([find_start_pos(input_ids[i], cat_ids_list[i])-1 for i in range(input_ids.size(0))], device=device)

    B, T, V = log_prob.shape
    max_cat_len = max(cat_lens).item()
    ar = torch.arange(max_cat_len, device=device)

    logit_positions = cat_logit_start[:, None] + ar[None, :]
    logit_positions = logit_positions.clamp(min=0, max=T-1)

    lp_slice = log_prob.gather(1, logit_positions[:, :, None].expand(B, max_cat_len, V))
    pred_slice = predictions.gather(1, logit_positions)

    cat_logprob = torch.zeros(B, device=device)
    pearson = torch.zeros(B, device=device)
    predictions_list = []
    for i in range(B):
        L = cat_lens[i]
        ids = cat_ids_list[i]
        cat_logprob[i] = lp_slice[i, :L, :].gather(1, ids[:, None]).mean().item()
        predictions_list.append(mm.tokenizer.decode(pred_slice[i, :L].tolist()))

        target = torch.tensor(mm.tokenizer(" " + data[i]["target"], add_special_tokens=False)["input_ids"], device=device)
        target_len = target.size(0)
        target_start = find_start_pos(input_ids[i], target)
        target_embedding = get_embedding(
            start_pos=target_start,
            end_pos=target_start+target_len,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i],
        )
        category_embedding = get_embedding(
            start_pos=cat_logit_start[i]+1,
            end_pos=cat_logit_start[i]+1+L,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i],
        )
        pearson[i] = torch.tensor(pearsonr(target_embedding, category_embedding)[0], device=device)
    
    cc_idx = torch.arange(0, B, 3, device=device)
    wc1_idx  = torch.arange(1, B, 3, device=device)
    wc2_idx  = torch.arange(2, B, 3, device=device)
    cc_lp = cat_logprob[cc_idx]
    wc1_lp = cat_logprob[wc1_idx]
    wc2_lp = cat_logprob[wc2_idx]
    cc_wc1_lp_correct = cc_lp > wc1_lp
    cc_wc2_lp_correct = cc_lp > wc2_lp

    target_cc_pearsonr = pearson[cc_idx]
    target_wc1_pearsonr = pearson[wc1_idx]
    target_wc2_pearsonr = pearson[wc2_idx]
    cc_wc1_embed_correct = target_cc_pearsonr > target_wc1_pearsonr
    cc_wc2_embed_correct = target_cc_pearsonr > target_wc2_pearsonr

    results = []
    for i in range(0, input_ids.shape[0], 3):
        results.append({
            "input": data[i]["input"],
            "cc": data[i]["category"],
            "wc1": data[i+1]["category"],
            "wc2": data[i+2]["category"],
            "cc_wc1_logprob": cc_wc1_lp_correct[i//3].item(),
            "cc_wc2_logprob": cc_wc2_lp_correct[i//3].item(),
            "cc_wc1_embed": cc_wc1_embed_correct[i//3].item(),
            "cc_wc2_embed": cc_wc2_embed_correct[i//3].item(),
            "preds": (predictions_list[i], predictions_list[i+1]),
        })

    num_examples = len(results)
    accuracy = sum(r["cc_wc2_logprob"] for r in results) / num_examples 
    wc1_accuracy = sum(r["cc_wc1_logprob"] for r in results) / num_examples 
    embed_accuracy = sum(r["cc_wc2_embed"] for r in results) / num_examples 
    wc1_embed_accuracy = sum(r["cc_wc1_embed"] for r in results) / num_examples 
    print(f"(wc1) lp accuracy: {wc1_accuracy:.3f}")
    print(f"(wc2) lp accuracy: {accuracy:.3f}")
    print(f"(wc1) embed accuracy: {wc1_embed_accuracy:.3f}")
    print(f"(wc2) embed accuracy: {embed_accuracy:.3f}")

    return results

def cat_eval_B(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([input.to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([prediction.to(device) for prediction in predictions], pad_value=-100)

    yes_ids = torch.tensor(
        mm.tokenizer(" " + "Yes", add_special_tokens=False)["input_ids"],
        device=device, dtype=torch.long
    )
    no_ids = torch.tensor(
        mm.tokenizer(" " + "No", add_special_tokens=False)["input_ids"],
        device=device, dtype=torch.long
    )
    assert len(yes_ids) == len(no_ids)
    L = len(yes_ids)

    log_prob = F.log_softmax(logits, dim=-1)

    input_lens = (input_ids != pad_id).sum(dim=1)
    start_idx = input_lens - L - 1  

    idx_range = torch.arange(L, device=device)
    token_positions = start_idx[:, None] + idx_range

    B, T, V = log_prob.shape
    lp_slice = log_prob.gather(1, token_positions[:, :, None].expand(B, L, V))
    pred_slice = predictions.gather(1, token_positions)
                                         
    yes_lp = lp_slice.gather(2, yes_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)
    no_lp = lp_slice.gather(2, no_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)

    # true_is_yes = torch.tensor([a.strip().lower() == "yes" for a in data["answer"]],device=device)
    # correct = torch.where(true_is_yes, yes_lp > no_lp, no_lp > yes_lp)

    saying_yes = yes_lp > no_lp
    cc_idx = torch.arange(0, B, 3, device=device)
    wc1_idx  = torch.arange(1, B, 3, device=device)
    wc2_idx  = torch.arange(2, B, 3, device=device)
    cc_yes_precent = saying_yes[cc_idx].float().mean().item()
    wc1_yes_percent = saying_yes[wc1_idx].float().mean().item()
    wc2_yes_percent = saying_yes[wc2_idx].float().mean().item()
    print(f"cc yes percent: {cc_yes_precent:.3f}")
    print(f"wc1 yes percent: {wc1_yes_percent:.3f}")
    print(f"wc2 yes percent: {wc2_yes_percent:.3f}")

    results = []
    for i in range(0, input_ids.shape[0], 3):
        results.append({
            "cc_input": data[i]["input"],
            "wc1_input": data[i+1]["input"],
            "wc2_input": data[i+2]["input"],
            "cc_yes_percent": saying_yes[i].item(),
            "wc1_yes_percent": saying_yes[i+1].item(),
            "wc2_yes_percent": saying_yes[i+2].item(),
            "preds": (mm.tokenizer.decode(pred_slice[i].tolist()),
                      mm.tokenizer.decode(pred_slice[i+1].tolist()),
                      mm.tokenizer.decode(pred_slice[i+2].tolist())),
        })
    return results
  
def cohypo_eval_A(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([input.to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([prediction.to(device) for prediction in predictions], pad_value=-100)
    embeddings = list(chain.from_iterable(embeds))

    log_prob = F.log_softmax(logits, dim=-1)

    cword_ids_list = [
        torch.tensor(mm.tokenizer(" " + d["c_word"], add_special_tokens=False)["input_ids"], device=device)
        for d in data
    ]

    input_lens = (input_ids != pad_id).sum(dim=1)
    cword_lens = torch.tensor([len(x) for x in cword_ids_list], device=device)
    cword_logit_start = torch.tensor([find_start_pos(input_ids[i], cword_ids_list[i])-1 for i in range(input_ids.size(0))], device=device)

    B, T, V = log_prob.shape
    max_cword_len = max(cword_lens).item()
    ar = torch.arange(max_cword_len, device=device)

    logit_positions = cword_logit_start[:, None] + ar[None, :]
    logit_positions = logit_positions.clamp(min=0, max=T-1)

    lp_slice = log_prob.gather(1, logit_positions[:, :, None].expand(B, max_cword_len, V))
    pred_slice = predictions.gather(1, logit_positions)

    cword_logprob = torch.zeros(B, device=device)
    pearson = torch.zeros(B, device=device)
    predictions_list = []
    for i in range(B):
        L = cword_lens[i]
        ids = cword_ids_list[i]
        cword_logprob[i] = lp_slice[i, :L, :].gather(1, ids[:, None]).mean().item()
        predictions_list.append(mm.tokenizer.decode(pred_slice[i, :L].tolist()))

        target = torch.tensor(mm.tokenizer(" " + data[i]["target"], add_special_tokens=False)["input_ids"], device=device)
        target_len = target.size(0)
        target_start = find_start_pos(input_ids[i], target)
        target_embedding = get_embedding(
            start_pos=target_start,
            end_pos=target_start+target_len,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i]
        )
        category_embedding = get_embedding(
            start_pos=cword_logit_start[i]+1,
            end_pos=cword_logit_start[i]+1+L,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i]
        )
        pearson[i] = torch.tensor(pearsonr(target_embedding, category_embedding)[0], device=device)
    
    c1_idx = torch.arange(0, B, 4, device=device)
    c2_idx = torch.arange(1, B, 4, device=device)
    c3_idx = torch.arange(2, B, 4, device=device)
    c4_idx = torch.arange(3, B, 4, device=device)
    c1_lp = cword_logprob[c1_idx]
    c2_lp = cword_logprob[c2_idx]
    c3_lp = cword_logprob[c3_idx]
    c4_lp = cword_logprob[c4_idx]

    c1_c2_correct = c1_lp > c2_lp
    c1_c3_correct = c1_lp > c3_lp
    c1_c4_correct = c1_lp > c4_lp

    target_c1_pearsonr = pearson[c1_idx]
    target_c2_pearsonr = pearson[c2_idx]
    target_c3_pearsonr = pearson[c3_idx]
    target_c4_pearsonr = pearson[c4_idx]

    c1_c2_embed_correct = target_c1_pearsonr > target_c2_pearsonr
    c1_c3_embed_correct = target_c1_pearsonr > target_c3_pearsonr
    c1_c4_embed_correct = target_c1_pearsonr > target_c4_pearsonr


    results = []
    for i in range(0, B, 4):
        results.append({
            "input": data[i]["input"],
            "c1": data[i]["c_word"],
            "c2": data[i+1]["c_word"],
            "c3": data[i+2]["c_word"],
            "c4": data[i+3]["c_word"],
            "c1_c2_logprob": c1_c2_correct[i//4].item(),
            "c1_c3_logprob": c1_c3_correct[i//4].item(),
            "c1_c4_logprob": c1_c4_correct[i//4].item(),
            "c1_c2_embed": c1_c2_embed_correct[i//4].item(),
            "c1_c3_embed": c1_c3_embed_correct[i//4].item(),
            "c1_c4_embed": c1_c4_embed_correct[i//4].item(),
            "preds": (predictions_list[i], predictions_list[i+1], predictions_list[i+2], predictions_list[i+3]),
        })
        
    num_examples = len(results)
    c1_c2_accuracy = sum(r["c1_c2_logprob"] for r in results) / num_examples
    c1_c2_embed_accuracy = sum(r["c1_c2_embed"] for r in results) / num_examples              
    c1_c3_accuracy = sum(r["c1_c3_logprob"] for r in results) / num_examples
    c1_c3_embed_accuracy = sum(r["c1_c3_embed"] for r in results) / num_examples
    c1_c4_accuracy = sum(r["c1_c4_logprob"] for r in results) / num_examples
    c1_c4_embed_accuracy = sum(r["c1_c4_embed"] for r in results) / num_examples

    print(f"(c1_c2) lp accuracy: {c1_c2_accuracy:.3f}")
    print(f"(c1_c3) lp accuracy: {c1_c3_accuracy:.3f}")
    print(f"(c1_c4) lp accuracy: {c1_c4_accuracy:.3f}")
    print(f"(c1_c2) embed accuracy: {c1_c2_embed_accuracy:.3f}")
    print(f"(c1_c3) embed accuracy: {c1_c3_embed_accuracy:.3f}")
    print(f"(c1_c4) embed accuracy: {c1_c4_embed_accuracy:.3f}")
    return results

def cohypo_eval_B(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([logit.to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([input.to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([prediction.to(device) for prediction in predictions], pad_value=-100)

    yes_ids = torch.tensor(
        mm.tokenizer(" Yes", add_special_tokens=False)["input_ids"],
        device=device, dtype=torch.long
    )
    no_ids = torch.tensor(
        mm.tokenizer(" No", add_special_tokens=False)["input_ids"],
        device=device, dtype=torch.long
    )
    assert len(yes_ids) == len(no_ids)
    L = len(yes_ids)

    log_prob = F.log_softmax(logits, dim=-1)

    input_lens = (input_ids != pad_id).sum(dim=1)
    start_idx = input_lens - L - 1  

    idx_range = torch.arange(L, device=device)
    token_positions = start_idx[:, None] + idx_range

    B, T, V = log_prob.shape
    lp_slice = log_prob.gather(1, token_positions[:, :, None].expand(B, L, V))
    pred_slice = predictions.gather(1, token_positions)
                                         
    yes_lp = lp_slice.gather(2, yes_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)
    no_lp = lp_slice.gather(2, no_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)

    # true_is_yes = torch.tensor([a.strip().lower() == "yes" for a in data["answer"]],device=device)

    # correct = torch.where(true_is_yes, yes_lp > no_lp, no_lp > yes_lp)
    saying_yes = yes_lp > no_lp
    c1_idx = torch.arange(0, B, 4, device=device)
    c2_idx  = torch.arange(1, B, 4, device=device)
    c3_idx  = torch.arange(2, B, 4, device=device)
    c4_idx  = torch.arange(3, B, 4, device=device)
    c1_yes_precent = saying_yes[c1_idx].float().mean().item()
    c2_yes_precent = saying_yes[c2_idx].float().mean().item()
    c3_yes_precent = saying_yes[c3_idx].float().mean().item()
    c4_yes_precent = saying_yes[c4_idx].float().mean().item()
    print(f"c1 yes precentage: {c1_yes_precent:.3f}")
    print(f"c2 yes precentage: {c2_yes_precent:.3f}")
    print(f"c3 yes precentage: {c3_yes_precent:.3f}")
    print(f"c4 yes precentage: {c4_yes_precent:.3f}")
    results = []
    for i in range(0, input_ids.shape[0], 4):
        results.append({
            "c1_input": data[i]["input"],
            "c2_input": data[i+1]["input"],
            "c3_input": data[i+2]["input"],
            "c4_input": data[i+3]["input"],
            "c1_yes_percent": saying_yes[i].item(),
            "c2_yes_percent": saying_yes[i+1].item(),
            "c3_yes_percent": saying_yes[i+2].item(),
            "c4_yes_percent": saying_yes[i+3].item(),
            "preds": (mm.tokenizer.decode(pred_slice[i].tolist()),
                      mm.tokenizer.decode(pred_slice[i+1].tolist()),
                      mm.tokenizer.decode(pred_slice[i+2].tolist()),
                      mm.tokenizer.decode(pred_slice[i+3].tolist())),
        })
    return results

    