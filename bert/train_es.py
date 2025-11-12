# train_es.py
import argparse
import time
import torch
import copy
import numpy as np
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from utils import load_kaggle_imdb, dataset_to_dataloader, compute_metrics, start_nvml_logger, save_results
from tqdm import tqdm

def get_adapter_params(model):
    # return flattened parameter vector and a list of param tensors to map back
    params = []
    shapes = []
    names = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p.detach().cpu().numpy().ravel())
            shapes.append(p.size())
            names.append(n)
    flat = np.concatenate(params).astype(np.float32)
    return flat, params, shapes, names

def set_adapter_params(model, flat, names, shapes):
    # write flat back into model.params (assume same ordering)
    idx = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            size = int(np.prod(p.size()))
            arr = flat[idx:idx+size].reshape(p.size())
            p.data.copy_(torch.from_numpy(arr).to(p.device))
            idx += size

def eval_fitness(model, val_loader, device='cuda'):
    model.eval()
    preds = []
    labs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
            preds.extend(probs.tolist())
            labs.extend(labels.detach().cpu().numpy().tolist())
    metrics = compute_metrics(np.array(preds), np.array(labs))
    # use F1 + AUROC as combined fitness or negative loss; here we'll use AUROC + F1
    return metrics, metrics.get('auroc', 0.0) + metrics.get('f1', 0.0)

def train_es(model_name='bert-base-uncased', batch_size=16, generations=100, pop_size=30, sigma=0.02, alpha=0.1, epochs=1, output_dir='es_output'):
    tokenized, tokenizer = load_kaggle_imdb(csv_path='imdb_dataset.csv', tokenizer_name=model_name, max_length=256)
    train_loader = dataset_to_dataloader(tokenized, 'train', batch_size=batch_size, shuffle=True)
    val_loader = dataset_to_dataloader(tokenized, 'test', batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    #print(f"Using device: {device}")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # attach LoRA adapter and keep its params as the ones we will optimize via ES
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # collect adapter params as a flat numpy vector
    flat0, parts, shapes, names = get_adapter_params(model)
    flat = flat0.copy()
    print("Adapter param vector length:", len(flat))

    nvml_stop = start_nvml_logger(outpath='es_gpu_log.csv', interval=1.0)

    results = {'generations':[], 'best_metrics': None}
    best_flat = flat.copy()
    best_score = -1e9
    t0 = time.time()
    for gen in range(generations):
        # sample population of perturbations
        noises = np.random.randn(pop_size, flat.shape[0]).astype(np.float32)
        rewards = np.zeros(pop_size, dtype=np.float32)
        # Evaluate population members
        for i in range(pop_size):
            candidate = flat + sigma * noises[i]
            # set into model
            set_adapter_params(model, candidate, names, shapes)
            # quick fine-tune step? or direct eval: do a quick eval on a subset of validation to speed up
            metrics, score = eval_fitness(model, val_loader, device='cuda')
            rewards[i] = score
        # normalize rewards
        A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # NES update
        grad_estimate = np.dot(A, noises) / pop_size
        flat = flat + alpha * grad_estimate
        # evaluate new flat
        set_adapter_params(model, flat, names, shapes)
        metrics, score = eval_fitness(model, val_loader, device='cuda')
        results['generations'].append({'gen':gen, 'score': float(score), 'metrics': metrics})
        print(f"[ES] Gen {gen}: score={score:.4f} metrics={metrics}")
        if score > best_score:
            best_score = score
            best_flat = flat.copy()
            results['best_metrics'] = {'gen':gen, 'score': score, 'metrics': metrics}
    total_time = time.time() - t0
    results['total_time_s'] = total_time
    nvml_stop.set()
    # save best weights into model and save
    set_adapter_params(model, best_flat, names, shapes)
    model.save_pretrained(output_dir)
    save_results(output_dir + '/es_results.json', results)
    print("Saved ES model and results.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--output_dir', default='es_output')
    args = parser.parse_args()
    train_es(model_name=args.model_name, batch_size=args.batch_size, generations=args.generations, pop_size=args.pop_size, sigma=args.sigma, alpha=args.alpha, output_dir=args.output_dir)
