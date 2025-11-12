# eval_plot.py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_json(path):
    with open(path,'r') as f:
        return json.load(f)

def plot_metrics(lora_path, es_path, outdir='plots'):
    l = load_json(lora_path)
    e = load_json(es_path)

    # extract time series
    l_epochs = l.get('train_steps', [])
    e_gens = e.get('generations', [])

    # prepare dataframes
    l_df = pd.DataFrame([{'epoch':s['epoch'], 'avg_loss': s['avg_loss'], 'time_s': s['time_s']} for s in l_epochs])
    e_df = pd.DataFrame([{'gen':s['gen'], 'score': s['score'], 'time_s': None} for s in e_gens])

    # validation metrics
    l_val = pd.DataFrame(l.get('val_metrics', []))
    e_val = pd.DataFrame([g['metrics'] for g in e_gens])

    # plots
    plt.figure(figsize=(8,5))
    if not l_val.empty:
        plt.plot(l_val['accuracy'], label='LoRA Accuracy')
    if not e_val.empty:
        plt.plot(e_val['accuracy'], label='ES Accuracy')
    plt.xlabel('Epoch / Generation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Comparison')
    plt.savefig(outdir + '/accuracy_compare.png')
    plt.close()

    # F1
    plt.figure(figsize=(8,5))
    if not l_val.empty:
        plt.plot(l_val['f1'], label='LoRA F1')
    if not e_val.empty:
        plt.plot(e_val['f1'], label='ES F1')
    plt.xlabel('Epoch / Generation')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Comparison')
    plt.savefig(outdir + '/f1_compare.png')
    plt.close()

    # print summary
    print("LoRA final metrics:", l.get('val_metrics', [])[-1] if l.get('val_metrics') else None)
    print("ES best metrics:", e.get('best_metrics'))
