from importlib.resources import path
import re
import os
import random
import json
import copy
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from psychai.language import load_any_as_chat
from psychai.language.tokenizer import make_normalizer, make_pretokenizer, train_tokenizer, wrap_tokenizer, print_tokenizer

def prepare_childes_data():
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

def create_raw_data(path="./data/ACL/LLM_Categories_stim.xlsx"):
    df_dict = pd.read_excel(path, sheet_name=None)
    category_dict = dict(zip(df_dict["probes"]["instance"], df_dict["probes"]["category"]))

    raw_df = copy.deepcopy(df_dict["cohyponyms"])
    cohypo_cols = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
    for i, c in enumerate(cohypo_cols, start=1):
        raw_df[f"category{i}"] = raw_df[c].map(category_dict)

    raw_df.to_json("./data/ACL/stimuli_with_categories.jsonl", orient="records", lines=True)

def prepare_evaluation_data(eval_type: str = "base", 
                            category_file: str = "./data/ACL/stimuli_with_categories.jsonl",
                            raw_file: str = "./data/ACL/LLM_Categories_stim.xlsx",
                            output_dir: str = "./data/ACL/plain_eval_data"):
    records = []
    with open(category_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df_dict = pd.read_excel(raw_file, sheet_name=None)
    probe_determiner_dict = dict(zip(df_dict["probes"]["instance"], df_dict["probes"]["probe_determiner"].fillna("")))
    category_determiner_dict = dict(zip(df_dict["categories"]["category"], df_dict["categories"]["category_determiner"].fillna("")))

    os.makedirs(output_dir, exist_ok=True)

    def superordinate():
        metaprompts = {"task specific": "Please complete the following sentence about the category label for the word that is provided. Respond as concisely as possible. ",
                       "neutral": "Please complete the following sentence in a natural and fluent way in English. Respond as concisely as possible. ",
                       "none": " "}

        prompt_templates = {
                "task specific 1": "{W} {X} is {Y}",
                "task specific 2": "{W} {X} is a kind of",
                "task specific 3": "{W} {X} is a type of",
                "task specific 4": "{W} {X} belongs to the category",
                "task specific 5": "{W} {X} is classified as {Y}",
                "control 1": "{X}",
                "control 2": "{X}:",
                "control 3": "{X} ->",
                "control 4": "{X} —",
                "control 5": "{X} and"
        }
        path = f"{output_dir}/superordinate.jsonl"
        conditions = ["category1", "category3", "category4"]
        with open(path, "w", encoding="utf-8") as f:
            for metaprompt_type, metaprompt in metaprompts.items():
                for prompt_type, prompt_template in prompt_templates.items():
                    for rec in records:
                        for i in range(len(conditions)):
                            probe = rec["probe"]
                            probe_determiner = probe_determiner_dict[probe]
                            category = rec[conditions[i]]
                            category_determiner = category_determiner_dict[category]

                            target = " " + category
                            if eval_type == "base":
                                input_text = prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner).strip() + f"{target}."
                            else:
                                input_text = [
                                    {"role": "user", "content": metaprompt + prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner).strip()},
                                    {"role": "assistant", "content": f"{target}."}
                                    ],

                            f.write(json.dumps({
                                "relationship": "superordinate",
                                "task": "cloze",
                                "condition": conditions[i],
                                "meta_prompt_type": metaprompt_type,
                                "prompt_type": prompt_type,
                                "input_text": metaprompt + input_text,
                                "probe": " " + probe,
                                "comparison": category,
                                "target": target,
                            }, ensure_ascii=False) + "\n")

    def cohyponym():
        metaprompts = {"task specific": "Please complete the following sentence about words and whether they belong to the same category. Respond as concisely as possible. ",
                       "neutral": "Please complete the following sentence in a natural and fluent way in English. Respond as concisely as possible. ",
                       "none": " "}

        prompt_templates = {
                "task specific 1": "{W} {X} is like {Y}",
                "task specific 2": "{W} {X} is similar to {Y}",
                "task specific 3": "Two words that belong to the same category are {X} and",
                "task specific 4": "Another word that belongs to the same category as {X} is",
                "task specific 5": "{X} is the same type of thing as",
                "control 1": "{X}",
                "control 2": "{X}:",
                "control 3": "{X} ->",
                "control 4": "{X} —",
                "control 5": "{X} and"
        }
        
        conditions = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
        path = f"{output_dir}/cohyponym.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for metaprompt_type, metaprompt in metaprompts.items():
                for prompt_type, prompt_template in prompt_templates.items():
                    for rec in records:
                        for i in range(len(conditions)):
                            probe = rec["probe"]
                            probe_determiner = probe_determiner_dict[probe]
                            cohypo = rec[conditions[i]]
                            cohyp_determiner = probe_determiner_dict[cohypo]

                            target = " " + cohypo
                            if eval_type == "base":
                                input_text = prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner).strip() + f"{target}."
                            else:
                                input_text = [
                                    {"role": "user", "content": metaprompt + prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner).strip()},
                                    {"role": "assistant", "content": f"{target}."}
                                    ],

                            f.write(json.dumps({
                                "relationship": "cohyponym",
                                "task": "cloze",
                                "condition": conditions[i],
                                "meta_prompt_type": metaprompt_type,
                                "prompt_type": prompt_type,
                                "input_text": metaprompt + input_text,
                                "probe": " " + probe,
                                "comparison": cohypo,
                                "target": target,
                            }, ensure_ascii=False) + "\n")
    
    superordinate()
    cohyponym()

def prepare_evaluation_data_for_kara(path="/Users/jingfengzhang/Desktop/CLOC Stimuli.xlsx"):
    df_dict = pd.read_excel(path, sheet_name="Stimuli")
    df_dict = df_dict.drop(df_dict.index[:8])
    stimuli = df_dict[["SenID", "SentFrame", "Word"]]
    stimuli["Word"] = " " + stimuli["Word"]
    stimuli["full_sentence"] = stimuli["SentFrame"] + stimuli["Word"]
    record = {}
    with open("/Users/jingfengzhang/Desktop/cloc_stimuli.json", "w") as f:
        for _, row in stimuli.iterrows():
            record["input_text"] = row['full_sentence']
            record["target"] = row['Word']
            json.dump(record, f)
            f.write("\n")

def main():
    # create_raw_data()
    prepare_evaluation_data(eval_type="base",
                            category_file="./data/ACL/stimuli_with_categories.jsonl",
                            raw_file="./data/ACL/LLM_Categories_stim.xlsx",
                            output_dir="./data/ACL/plain_eval_data")

if __name__ == "__main__":
    main()