import pandas as pd
import pylangacq as pla
from pathlib import Path
from datasets import load_dataset
import pickle
from psychai.tokenizer.tokenizer import make_normalizer, make_pretokenizer, train_tokenizer, wrap_tokenizer, print_tokenizer

def main():
    # all_reader = pickle.load(open("./data/childes/eng_na_reader.pkl", "rb"))
    # word_freq = all_reader.word_frequencies()
    # word_freq_df = pd.DataFrame(word_freq.items(), columns=["word", "count"])
    # word_freq_df = word_freq_df.sort_values(by="count", ascending=False)
    # word_freq_df.to_csv("./data/childes/eng_na_vocab_counts.csv", index=False)

    docs_path = Path("./data/childes/text_docs")

    if docs_path.exists():
        documents = [str(f) for f in docs_path.rglob("*.txt")]
    else:
        reader = pickle.load(open("./data/childes/bates_reader.pkl", "rb"))
        words_by_files = reader.words(by_files=True, by_utterances=False, exclude="CHI")
        documents = []
        for i, words in enumerate(words_by_files):
            doc = " ".join(words)
            documents.append(doc)
            with open(docs_path / f"doc_{i}.txt", "w", encoding="utf-8") as f:
                f.write(doc)
            

    normalizer = make_normalizer(lowercase=True, 
                            strip=True, 
                            strip_left=True, 
                            strip_right=True, 
                            strip_accents=True)
    
    pretokenizer = make_pretokenizer(use_whitespace=True,
                                     use_punctuation=True,
                                     split_hyphens=True)
    

    
    tokenizer = train_tokenizer(files=documents, model_type="wordlevel", normalizer=normalizer, pretokenizer=pretokenizer)
    tokenizer = wrap_tokenizer(tokenizer, bos_token="<bos>", eos_token="<eos>")
    tokenizer.save_pretrained("./tokenizer/full_bates_tokenizer")

    

if __name__ == "__main__":
    main()