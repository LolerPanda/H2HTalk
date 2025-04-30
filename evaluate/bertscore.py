#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERTScore module, provides text similarity scoring based on BERT
"""
import os   
import time
import torch
import gc  # Add garbage collection module
from core import timed_print, HAVE_LOCAL_BERTSCORE_MODEL, HAVE_BERT_SCORE, LOCAL_MODEL_DIR, BERTSCORE_MODEL_NAME


class BertModelCache:
    _instance = None

    def __init__(self):
        if BertModelCache._instance is not None:
            raise Exception("BertModelCache is a singleton, please use instance() method to get the instance.")
        # Load model, tokenizer and device using existing loading function
        self.tokenizer, self.model, self.device = load_bert_model()
        # If bert_score library supports internal caching, register the loaded model to internal cache
        try:
            import bert_score
            bert_score._models["bert-base-chinese"] = (self.tokenizer, self.model, self.device)
        except Exception:
            pass
        BertModelCache._instance = self

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BertModelCache()
        return cls._instance
    
    @classmethod
    def clear_cache(cls):
        """Clear CUDA cache and release tensor memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def calculate_bertscore(hypothesis, reference, silent=False):
    """
    Calculate BERTScore using pre-loaded model to avoid repeated loading
    """
    try:
        # Check if input is empty
        if not hypothesis or not reference:
            if not silent:
                timed_print("Warning: BERTScore calculation received empty input")
            return 0.0

        # Ensure input is string
        hypothesis = str(hypothesis)
        reference = str(reference)

        start_time = time.time()

        # Get cached model, tokenizer and device through singleton pattern
        cache = BertModelCache.instance()
        if not silent:
            timed_print(f"BERTScore model cache info: device={cache.device}, model loaded={cache.model is not None}, tokenizer loaded={cache.tokenizer is not None}")
            
        device = cache.device

        import bert_score
        # Define calculation parameters to ensure bert_score can use cached model
        calc_params = {
            "model_type": "bert-base-chinese",
            "lang": "zh",
            "verbose": False,
            "device": device,
            "rescale_with_baseline": False,
        }

        # Limit input length (e.g. to prevent memory overflow)
        max_length = 128
        if len(hypothesis) > max_length or len(reference) > max_length:
            if not silent:
                timed_print(f"BERTScore: Input length exceeds limit, truncating to {max_length} characters")
            hypothesis = hypothesis[:max_length]
            reference = reference[:max_length]

        # Use torch.no_grad() context manager to disable gradient calculation
        with torch.no_grad():
            # Call bert_score.score() to calculate score, it will detect _models cache
            P, R, F1 = bert_score.score(
                [hypothesis],
                [reference],
                **calc_params
            )
            score = F1.item()

            # Explicitly release tensors that are no longer needed
            # del P, R, F1
            
        # Clean cache after calculation
       
        # Force garbage collection
        # gc.collect()
            
        calc_time = time.time() - start_time
        if calc_time > 1.0 and not silent:
            timed_print(f"BERTScore calculation took longer: {calc_time:.2f} seconds - Input length: {len(hypothesis)}/{len(reference)}")
        BertModelCache.clear_cache()
        gc.collect()
            
        return score

    except Exception as e:
        if not silent:
            timed_print(f"Error calculating BERTScore: {str(e)}")
        # Clean cache on error
        BertModelCache.clear_cache()
        gc.collect()
        return 0.0

def load_bert_model():
    """Load BERT model"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # Clean up any existing old CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if HAVE_LOCAL_BERTSCORE_MODEL:
            model_path = f"{LOCAL_MODEL_DIR}/{BERTSCORE_MODEL_NAME}"
            timed_print(f"Loading local BERT model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
        else:
            # Default to using Chinese BERT model
            model_name = "bert-base-chinese"
            timed_print(f"Loading remote BERT model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        
        # Move model to available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return tokenizer, model, device
    except Exception as e:
        timed_print(f"Failed to load BERT model: {e}")
        return None, None, "cpu" 