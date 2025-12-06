import json
import torch
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

def save_results(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved results to {out_path}")

def find_start_pos(seq, subseq):
    T = seq.size(0)
    L = subseq.size(0)
    for i in range(T - L + 1):
        if torch.equal(seq[i:i+L], subseq):
            return i
    return -1

def get_embedding(start_pos, end_pos, input_ids, embeds, layer_name, embed_type="embeddings"):
    embeddings = []
    token_ids = []
    for pos in range(start_pos, end_pos):
        assert embeds[pos]["token_id"] == input_ids[pos], "Token ID mismatch in embeddings"
        embeddings.append(embeds[pos]["layers"][layer_name][embed_type])
        token_ids.append(embeds[pos]["token_id"])
    embedding_mean = np.mean(embeddings, axis=0)
    return embedding_mean

def cat_eval_A(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}/eval_data/cat_eval_A.jsonl", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([torch.from_numpy(logit).to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([torch.from_numpy(input).to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([torch.from_numpy(prediction).to(device) for prediction in predictions], pad_value=-100)
    embeddings = list(chain.from_iterable(embeds))

    log_prob = F.log_softmax(logits, dim=-1)

    cat_ids_list = [
        torch.tensor(mm.tokenizer(" " + d["category"], add_special_tokens=False)["input_ids"], device=device)
        for d in data
    ]

    input_lens = (input_ids != pad_id).sum(dim=1)
    cat_lens = torch.tensor([len(x) for x in cat_ids_list], device=device)
    cat_logit_start = input_lens - cat_lens - 1

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
            layer_name=cfg.layer_type,
            embed_type=cfg.embed_type
        )
        category_embedding = get_embedding(
            start_pos=cat_logit_start[i]+1,
            end_pos=cat_logit_start[i]+1+L,
            input_ids=input_ids[i].tolist(),
            embeds=embeddings[i],
            layer_name=cfg.layer_type,
            embed_type=cfg.embed_type
        )
        pearson[i] = torch.tensor(pearsonr(target_embedding, category_embedding)[0], device=device)
    
    cc_idx = torch.arange(0, B, 2, device=device)
    wc_idx  = torch.arange(1, B, 2, device=device)
    cc_lp = cat_logprob[cc_idx]
    wc_lp = cat_logprob[wc_idx]
    lp_correct = cc_lp > wc_lp

    target_cc_pearsonr = pearson[cc_idx]
    target_wc_pearsonr = pearson[wc_idx]
    embed_correct = target_cc_pearsonr > target_wc_pearsonr

    results = []
    for i in range(0, input_ids.shape[0], 2):
        results.append({
            "input": data[i]["input"],
            "correct category": data[i]["category"],
            "wrong category": data[i+1]["category"],
            "cc_logprob": cc_lp[i//2].item(),
            "wc_logprob": wc_lp[i//2].item(),
            "correct": lp_correct[i//2].item(),
            "margin": (cc_lp[i//2] - wc_lp[i//2]).item(),
            "target_cc_pearsonr": target_cc_pearsonr[i//2].item(),
            "target_wc_pearsonr": target_wc_pearsonr[i//2].item(),
            "embed_correct": embed_correct[i//2].item(),
            "pearsonr_margin": (target_cc_pearsonr[i//2] - target_wc_pearsonr[i//2]).item(),
            "predictions": (predictions_list[i], predictions_list[i+1]),
        })

    num_examples = len(results)
    accuracy = sum(r["correct"] for r in results) / num_examples if num_examples > 0 else 0.0
    embed_accuracy = sum(r["embed_correct"] for r in results) / num_examples if num_examples > 0 else 0.0
    print(f"cat_eval_A accuracy: {accuracy:.3f} over {num_examples} examples")
    print(f"cat_eval_C accuracy: {embed_accuracy:.3f} over {num_examples} examples")
    save_results(results, f"{cfg.exp_dir}/cat_eval_A_results.jsonl")

    return {"accuracy": accuracy}   

def cat_eval_B(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}/eval_data/cat_eval_B.jsonl", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([torch.from_numpy(logit).to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([torch.from_numpy(input).to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([torch.from_numpy(prediction).to(device) for prediction in predictions], pad_value=-100)

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
                                         
    yes_lp = lp_slice.gather(2, yes_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)   # → [B]
    no_lp = lp_slice.gather(2, no_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)   # → [B]

    true_is_yes = torch.tensor([a.strip().lower() == "yes" for a in data["answer"]],device=device)

    correct = torch.where(true_is_yes, yes_lp > no_lp, no_lp > yes_lp)
    margin = yes_lp - no_lp
    accuracy = correct.float().mean().item()
    print(f"cat_eval_B yes/no accuracy: {accuracy:.3f} over {len(data['answer'])} examples")

    results = []
    for i in range(input_ids.shape[0]):
        results.append({
            "input": data[i]["input"],
            "true answer": data[i]["answer"],
            "yes_logprob": yes_lp[i].item(),
            "no_logprob": no_lp[i].item(),
            "correct": correct[i].item(),
            "margin": margin[i].item(),
            "predictions": mm.tokenizer.decode(pred_slice[i].tolist()),
        })
    save_results(results, f"{cfg.exp_dir}/cat_eval_B_results.jsonl")
    return {"accuracy": accuracy}
  
def cohypo_eval_A(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
        data = load_dataset("json", data_files=f"{cfg.data.test_path}/eval_data/cohypo_eval_A.jsonl", split="train")
        device = "cpu"
        pad_id = mm.tokenizer.pad_token_id

        logits = pad_and_concat([torch.from_numpy(logit).to(device) for logit in logits], pad_value=0.0)
        input_ids = pad_and_concat([torch.from_numpy(input).to(device) for input in inputs], pad_value=pad_id)
        predictions = pad_and_concat([torch.from_numpy(prediction).to(device) for prediction in predictions], pad_value=-100)
        embeddings = list(chain.from_iterable(embeds))

        log_prob = F.log_softmax(logits, dim=-1)

        cword_ids_list = [
            torch.tensor(mm.tokenizer(" " + d["c_word"], add_special_tokens=False)["input_ids"], device=device)
            for d in data
        ]

        input_lens = (input_ids != pad_id).sum(dim=1)
        cword_lens = torch.tensor([len(x) for x in cword_ids_list], device=device)
        cword_logit_start = input_lens - cword_lens - 1

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
                embeds=embeddings[i],
                layer_name=cfg.layer_type,
                embed_type=cfg.embed_type
            )
            category_embedding = get_embedding(
                start_pos=cword_logit_start[i]+1,
                end_pos=cword_logit_start[i]+1+L,
                input_ids=input_ids[i].tolist(),
                embeds=embeddings[i],
                layer_name=cfg.layer_type,
                embed_type=cfg.embed_type
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
        lp_correct = c1_lp > c4_lp

        target_c1_pearsonr = pearson[c1_idx]
        target_c2_pearsonr = pearson[c2_idx]
        target_c3_pearsonr = pearson[c3_idx]
        target_c4_pearsonr = pearson[c4_idx]
        embed_correct = target_c1_pearsonr > target_c4_pearsonr


        results = []
        for i in range(0, B, 4):
            results.append({
                "input": data[i]["input"],
                "c1_word": data[i]["c_word"],
                "c2_word": data[i+1]["c_word"],
                "c3_word": data[i+2]["c_word"],
                "c4_word": data[i+3]["c_word"],
                "c1_logprob": c1_lp[i//4].item(),
                "c2_logprob": c2_lp[i//4].item(),
                "c3_logprob": c3_lp[i//4].item(),
                "c4_logprob": c4_lp[i//4].item(),
                "correct": lp_correct[i//4].item(),
                "margin": c1_lp[i//4].item() - c4_lp[i//4].item(),
                "target_c1_pearsonr": target_c1_pearsonr[i//4].item(),
                "target_c2_pearsonr": target_c2_pearsonr[i//4].item(),
                "target_c3_pearsonr": target_c3_pearsonr[i//4].item(),
                "target_c4_pearsonr": target_c4_pearsonr[i//4].item(),
                "embed_correct": embed_correct[i//4].item(),
                "pearsonr_margin": target_c1_pearsonr[i//4].item() - target_c4_pearsonr[i//4].item(),
                "predictions": (predictions_list[i], predictions_list[i+1], predictions_list[i+2], predictions_list[i+3]),
            })
        num_examples = len(results)
        accuracy = sum(r["correct"] for r in results) / num_examples if num_examples > 0 else 0.0
        embed_accuracy = sum(r["embed_correct"] for r in results) / num_examples if num_examples > 0 else 0.0               
        print(f"cohypo_eval_A accuracy: {accuracy:.3f} over {num_examples} examples")
        print(f"cohypo_eval_A embedding accuracy: {embed_accuracy:.3f} over {num_examples} examples")
        save_results(results, f"{cfg.exp_dir}/cohypo_eval_A_results.jsonl")
        return {"accuracy": accuracy}

def cohypo_eval_B(mm, cfg, inputs, labels, logits, predictions, embeds, weights):
    data = load_dataset("json", data_files=f"{cfg.data.test_path}/eval_data/cohypo_eval_B.jsonl", split="train")
    device = "cpu"
    pad_id = mm.tokenizer.pad_token_id

    logits = pad_and_concat([torch.from_numpy(logit).to(device) for logit in logits], pad_value=0.0)
    input_ids = pad_and_concat([torch.from_numpy(input).to(device) for input in inputs], pad_value=pad_id)
    predictions = pad_and_concat([torch.from_numpy(prediction).to(device) for prediction in predictions], pad_value=-100)

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
                                         
    yes_lp = lp_slice.gather(2, yes_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)   # → [B]
    no_lp = lp_slice.gather(2, no_ids[None, :, None].expand(B, L, 1)).squeeze(-1).mean(dim=1)   # → [B]

    true_is_yes = torch.tensor([a.strip().lower() == "yes" for a in data["answer"]],device=device)

    correct = torch.where(true_is_yes, yes_lp > no_lp, no_lp > yes_lp)
    margin = yes_lp - no_lp
    accuracy = correct.float().mean().item()
    print(f"cohypo_eval_B yes/no accuracy: {accuracy:.3f} over {len(data['answer'])} examples")

    results = []
    for i in range(input_ids.shape[0]):
        results.append({
            "input": data[i]["input"],
            "true answer": data[i]["answer"],
            "yes_logprob": yes_lp[i].item(),
            "no_logprob": no_lp[i].item(),
            "correct": correct[i].item(),
            "margin": margin[i].item(),
            "predictions": mm.tokenizer.decode(pred_slice[i].tolist()),
        })
    save_results(results, f"{cfg.exp_dir}/cohypo_eval_B_results.jsonl")
    return {"accuracy": accuracy}

    