# utils.py
import time
import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import subprocess
import threading

def load_kaggle_imdb(csv_path, tokenizer_name='bert-base-uncased', max_length=256, test_size=0.2, seed=42):
    # 1. Load CSV
    df = pd.read_csv(csv_path)

    # 2. Rename columns for compatibility
    df = df.rename(columns={'review': 'text', 'sentiment': 'label'})

    # 3. Map labels to integers
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})

    # 4. Convert to HuggingFace dataset
    ds = Dataset.from_pandas(df[['text','label']])

    # 5. Split into train/test
    dataset = ds.train_test_split(test_size=test_size, seed=seed)
    dataset = DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })

    # 6. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    
    tokenized = dataset.map(tokenize, batched=True)
    return tokenized, tokenizer

# Placeholder hook to load MathQA/GSM8K
def load_mathqa_sample(path=None, tokenizer=None, max_len=256):
    from datasets import Dataset
    data = [
        {"question":"What is 2+3?","answer":"5"},
        {"question":"If x=2 and y=3 what's x*y?","answer":"6"},
    ]
    ds = Dataset.from_list(data)
    if tokenizer:
        def tok(ex):
            return tokenizer(ex['question'], truncation=True, padding='max_length', max_length=max_len)
        ds = ds.map(tok)
        ds.set_format(type='torch', columns=['input_ids','attention_mask'])
    return ds

def compute_metrics(preds, labels):
    preds_bin = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, preds_bin)
    f1 = f1_score(labels, preds_bin, zero_division=0)
    try:
        auroc = roc_auc_score(labels, preds)
    except Exception:
        auroc = float('nan')
    return {"accuracy": acc, "f1": f1, "auroc": auroc}

# nvidia-smi poller (runs in background)
def start_nvml_logger(outpath='gpu_log.csv', interval=1.0):
    stop = threading.Event()
    def poll():
        with open(outpath, 'w') as f:
            f.write('timestamp,utilization,gpu_mem_used_mb,gpu_mem_total_mb\n')
            while not stop.is_set():
                try:
                    p = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'], capture_output=True, text=True)
                    line = p.stdout.strip().replace('\n',';')
                    parts = [s.strip() for s in line.split(';') if s.strip()]
                    if parts:
                        vals = parts[0].split(',')
                        util, used, total = [v.strip() for v in vals]
                        f.write(f"{time.time()},{util},{used},{total}\n")
                        f.flush()
                except Exception:
                    pass
                time.sleep(interval)
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    return stop

def save_results(path, results):
    with open(path,'w') as f:
        json.dump(results, f, indent=2)

def collate_batch(batch):
    return {
        'input_ids': torch.tensor([b['input_ids'] for b in batch], dtype=torch.long),
        'attention_mask': torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long),
        'labels': torch.tensor([b['label'] for b in batch], dtype=torch.long)
    }


def dataset_to_dataloader(dataset, split='train', batch_size=16, shuffle=True):
    ds = dataset[split]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
