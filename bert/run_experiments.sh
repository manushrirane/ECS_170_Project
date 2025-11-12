#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export MODEL=bert-base-uncased

# LoRA quick smoke
python train_lora.py --model_name $MODEL --epochs 1 --batch_size 16 --output_dir lora_output

# ES quick smoke (smaller pop & gens)
python train_es.py --model_name $MODEL --generations 5 --pop_size 10 --sigma 0.02 --alpha 0.05 --batch_size 16 --output_dir es_output

# Evaluate & plot
python eval_plot.py lora_output/lora_results.json es_output/es_results.json
