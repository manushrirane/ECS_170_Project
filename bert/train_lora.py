# train_lora.py
import argparse
import time
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from utils import load_kaggle_imdb, dataset_to_dataloader, compute_metrics, start_nvml_logger, save_results
import numpy as np
from tqdm import tqdm

def train_lora(model_name='bert-base-uncased', batch_size=16, epochs=3, lr=2e-5, output_dir='lora_output', device='cuda'):
    # load dataset and tokenizer (IMDB)
    tokenized, tokenizer = load_kaggle_imdb(csv_path='imdb_dataset.csv', tokenizer_name=model_name, max_length=256)
    num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # Configure LoRA on the model
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]  # typical for attention heads
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # simple training loop with AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = dataset_to_dataloader(tokenized, 'train', batch_size=batch_size, shuffle=True)
    val_loader = dataset_to_dataloader(tokenized, 'test', batch_size=batch_size, shuffle=False)

    # Start nvml logger
    nvml_stop = start_nvml_logger(outpath='lora_gpu_log.csv', interval=1.0)

    results = {'train_steps':[], 'val_metrics':[], 'timings':{}}
    start_all = time.time()
    global_step = 0
    peak_alloc_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        losses = []
        for batch in tqdm(train_loader, desc=f"LoRA Train Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            global_step += 1

        epoch_time = time.time() - epoch_start
        results['train_steps'].append({'epoch':epoch, 'avg_loss': float(np.mean(losses)), 'time_s': epoch_time})

        # validation
        model.eval()
        preds = []
        labs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='LoRA Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
                preds.extend(probs.tolist())
                labs.extend(labels.detach().cpu().numpy().tolist())
        metrics = compute_metrics(np.array(preds), np.array(labs))
        results['val_metrics'].append(metrics)
        print(f"[LoRA] Epoch {epoch} metrics: {metrics}")

    total_time = time.time() - start_all
    peak_alloc_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    results['timings'] = {'total_time_s': total_time, 'peak_alloc_bytes_before': int(peak_alloc_before), 'peak_alloc_bytes_after': int(peak_alloc_after)}
    nvml_stop.set()
    save_results(output_dir + '/lora_results.json', results)
    model.save_pretrained(output_dir)
    print("Saved LoRA model and results.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--output_dir', default='lora_output')
    args = parser.parse_args()
    train_lora(model_name=args.model_name, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, output_dir=args.output_dir)
