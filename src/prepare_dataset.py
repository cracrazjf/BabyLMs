import re
import os
import random
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from psychai.language import make_normalizer, make_pretokenizer, train_tokenizer, wrap_tokenizer, print_tokenizer

def prepare_training_data():
    text_docs_path = Path("./data/childes/text_docs")
    text_docs_path.mkdir(parents=True, exist_ok=True)
    text_documents = [str(f) for f in text_docs_path.rglob("*.txt")]
      
    jsonl_docs_path = Path("./data/childes/jsonl_docs")
    jsonl_docs_path.mkdir(parents=True, exist_ok=True)
    
    normalizer = make_normalizer(lowercase=True)

    def clean_childes_csv():
        terminator_map = {
            "p": ".",
            "q": "?",
            "e": "!",
            "trail off": "…",
            "trail off question": "… ?",
            "interruption": "<i>",
            "self interruption": "<i>",
            "interruption question": "<i>",
            "self interruption question": "<i>",
            "quotation precedes": "<q>",
            "quotation next line": "<q>",
            "broken for coding": "<b>",
        }

        df = pd.read_csv("./data/childes/all_utterances.csv")
        df = df.dropna(subset=["utterance_text"])
        df = df[df.speaker_role != "Target_Child"]
        term_nan_df = df[df["terminator_type"].isna()]
        
        df["terminator_symbol"] = df["terminator_type"].map(terminator_map).fillna("")
        df["sentence_text"] = df["utterance_text"].astype(str) + " " + df["terminator_symbol"]

        grouped_df = df.groupby(["transcript"], as_index=False).agg(target_child_age=("target_child_age", "first"), text=("sentence_text", " ".join))
        grouped_term_nan_df = term_nan_df.groupby(["transcript"], as_index=False).agg(target_child_age=("target_child_age", "first"), text=("utterance_text", " ".join))    

        def clean_filename(s):
            return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(s))

        global_counter = Counter()
        na_counter = Counter()
        for _, row in grouped_df.iterrows():
            transcript = row["transcript"]
            age = row["target_child_age"]
            text = row["text"]

            normalized_text = normalizer.normalize_str(text)

            global_counter.update(normalized_text.split())

            base_name = clean_filename(transcript)
            json_fname = f"{jsonl_docs_path}/{base_name}.jsonl"

            with open(json_fname, "w", encoding="utf-8") as f:
                json_line = json.dumps({
                    "transcript": transcript,
                    "target_child_age": age,
                    "text": text
                }, ensure_ascii=False)
                f.write(json_line + "\n")

            txt_fname = f"{text_docs_path}/{base_name}.txt"
            with open(txt_fname, "w", encoding="utf-8") as f:
                f.write(text)

        for _, row in grouped_term_nan_df.iterrows():
            text = row["text"]
            normalized_text = normalizer.normalize_str(text)
            na_counter.update(normalized_text.split())

        nan_freq_df = pd.DataFrame(na_counter.items(), columns=["word", "frequency"])
        nan_freq_df = nan_freq_df.sort_values("frequency", ascending=False)
        nan_freq_df.to_csv("./data/childes/token_frequencies_nan_terminator.csv", index=False)

        freq_df = pd.DataFrame(global_counter.items(), columns=["word", "frequency"])
        freq_df = freq_df.sort_values("frequency", ascending=False)
        print("Total unique tokens:", len(freq_df))
        freq_df.to_csv("./data/childes/token_frequencies.csv", index=False)

    # if len(text_documents) == 0:
    clean_childes_csv()
    text_documents = [str(f) for f in text_docs_path.rglob("*.txt")]

    try:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer/childes_tokenizer")
    except:
        # pretokenizer = make_pretokenizer(use_whitespace=True)
        pretokenizer = make_pretokenizer(custom_splits=[(" ", "removed")])
        tokenizer = train_tokenizer(files=text_documents, 
                                    model_type="wordlevel", 
                                    normalizer=normalizer, 
                                    pretokenizer=pretokenizer,)
        
        tokenizer = wrap_tokenizer(tokenizer)
        print_tokenizer(tokenizer)
        tokenizer.save_pretrained("./tokenizer/childes_tokenizer")

    token_df = pd.DataFrame(tokenizer.get_vocab().items(), columns=["word", "id"])
    token_df = token_df.sort_values("word")
    token_df.to_csv("./tokenizer/childes_tokenizer/token_ids.csv", index=False)

def create_counterbalance_data(path="./data/childes/flores_stimuli.xlsx"):
    df = pd.read_excel(path)

    category_dict = dict(zip(df["Target"], df["Category"]))

    def create_wrongcat_dict(df):
        df["is_multi"] = df["Category"].str.split().str.len() > 1
        wrongcat = {}

        for is_multi, group in df.groupby("is_multi"):
            targets = group["Target"].tolist()
            cats = group["Category"].tolist()
            unique = sorted(group["Category"].unique().tolist())
            if len(unique) < 2:
                raise ValueError(
                    f"Group {'multi-word' if is_multi else 'single-word'} has fewer than "
                    f"2 unique categories; cannot counterbalance within this group."
                )
            tgt2cat = dict(zip(targets, cats))

            wrong_pool = {
                tgt: [c for c in unique if c != tgt2cat[tgt]]
                for tgt in targets
            }

            flat = []
            for tgt in targets:
                flat.extend(wrong_pool[tgt])

            random.shuffle(flat)

            remaining = flat.copy()

            for tgt in targets:
                true_cat = tgt2cat[tgt]
                for wc in remaining:
                    if wc != true_cat:
                        wrongcat[tgt] = wc
                        remaining.remove(wc)
                        break
        return wrongcat

    wrongcat_dict = create_wrongcat_dict(df)
    df["WrongCategory"] = df["Target"].map(wrongcat_dict)

    cols = ["Category", "Target"]
    for c in ["C1", "C2", "C3", "C4"]:
        if c in df.columns:
            cols.append(c)
    cols.append("WrongCategory")

    dataset_df = df[cols]

    jsonl_path = "./data/childes/flores_stimuli_with_wrongcat.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in dataset_df.iterrows():
            rec = {col: row[col] for col in cols}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Saved:", jsonl_path)

def prepare_evaluation_data(eval_type: str):
    records = []
    try:
        with open("./data/childes/flores_stimuli_with_wrongcat.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except:
        create_counterbalance_data()

    def _a_or_an(noun: str) -> str:
        mass_nones = ['makeup', 'toothpaste', 'lotion', 'drums', 'pliers', 'chalk']
        if noun in mass_nones:
            return ""
        else:
            vowels = "aeiou"
            noun = noun.strip().lower()

            if len(noun) == 0:
                return "a"

            return "an" if noun[0] in vowels else "a"
        
    def phrase(noun):
        article = _a_or_an(noun)
        return f"{article} {noun}".strip()

    os.makedirs("./data/childes/eval_data", exist_ok=True)

    if eval_type == "cat_eval_A":
        path = "./data/childes/eval_data/cat_eval_A.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                target = rec["Target"]
                correct_cat = rec["Category"]
                wrong_cat = rec["WrongCategory"]

                prompt = "Think of the category this object naturally belongs to. "

                for i in range(2):
                    if i == 0:
                        input_text = f"{phrase(target)} is {phrase(correct_cat)}"
                        f.write(json.dumps({
                            "prompt": prompt,
                            "input": input_text,
                            "target": target,
                            "category": correct_cat,
                            "correct": True
                        }, ensure_ascii=False) + "\n")
                    else:
                        input_text = f"{phrase(target)} is {phrase(wrong_cat)}"
                        f.write(json.dumps({
                            "prompt": prompt,
                            "input": input_text,
                            "target": target,
                            "category": wrong_cat,
                            "correct": False
                        }, ensure_ascii=False) + "\n")

    if eval_type == "cat_eval_B":
        path = "./data/childes/eval_data/cat_eval_B.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                target = rec["Target"]
                correct_cat = rec["Category"]
                wrong_cat = rec["WrongCategory"]

                prompt = "Answer the following question with Yes or No. "

                for i in range(2):
                    if i == 0:
                        input_text = f"Question: Is {phrase(target)} a type of {correct_cat}? Answer: Yes"
                        answer = "Yes"
                    else:
                        input_text = f"Question: Is {phrase(target)} a type of {wrong_cat}? Answer: No"
                        answer = "No"
                    f.write(json.dumps({
                        "prompt": prompt,
                        "input": input_text,
                        "answer": answer
                    }, ensure_ascii=False) + "\n")
    
    if eval_type == "cohypo_eval_A":
        path = "./data/childes/eval_data/cohypo_eval_A.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                target = rec["Target"]
                cs = [rec["C1"], rec["C2"], rec["C3"], rec["C4"]]

                prompt = "Think of an object that is semantically similar. "
                
                for i in range(4):
                    input_text = f"{phrase(target)} is like {phrase(cs[i])}"
                    f.write(json.dumps({
                        "prompt": prompt,
                        "input": input_text,
                        "target": target,
                        "c_word": cs[i],
                        "c": i+1,
                    }, ensure_ascii=False) + "\n")
                    
    if eval_type == "cohypo_eval_B":
        path = "./data/childes/eval_data/cohypo_eval_B.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                target = rec["Target"]
                cs = [rec["C1"], rec["C2"], rec["C3"], rec["C4"]]

                prompt = "Answer the following question with Yes or No. "

                for i in range(4):
                    if i < 2:
                        input_text = f"Question: Is {phrase(target)} like {phrase(cs[i])}? Answer: Yes"
                        answer = "Yes"
                    else:
                        input_text = f"Question: Is {phrase(target)} like {phrase(cs[i])}? Answer: No"
                        answer = "No"

                    f.write(json.dumps({
                        "prompt": prompt,
                        "input": input_text,
                        "answer": answer
                    }, ensure_ascii=False) + "\n")
