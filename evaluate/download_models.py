#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model download script
Download m3e-base and bert-base-chinese models required for evaluation
"""

import os
import argparse
from transformers import AutoModel, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Download models required for evaluation")
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'm3e-base', 'bert-base-chinese'],
                        help='Specify which model to download, default is all')
    args = parser.parse_args()
    
    # Model configuration
    models = {
        "m3e-base": "moka-ai/m3e-base",
        "bert-base-chinese": "bert-base-chinese"
    }
    
    # Determine which models to download
    models_to_download = []
    if args.model == 'all':
        models_to_download = list(models.keys())
    else:
        models_to_download = [args.model]
    
    # Create models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download models
    for model_name in models_to_download:
        model_path = os.path.join(models_dir, model_name)
        
        # Check if model already exists
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"Model {model_name} already exists at {model_path}")
            continue
        
        print(f"Starting download of model: {model_name}")
        try:
            # Download model and tokenizer
            model = AutoModel.from_pretrained(models[model_name])
            tokenizer = AutoTokenizer.from_pretrained(models[model_name])
            
            # Save to local
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            print(f"Model {model_name} downloaded successfully, saved to {model_path}")
        except Exception as e:
            print(f"Error downloading model {model_name}: {str(e)}")

    print("All models downloaded successfully")
    print(f"\nTo use local models for evaluation, run:")
    print(f"python simple_metrics_eval.py -d <model output directory> -r <reference answer directory>")

if __name__ == "__main__":
    main() 