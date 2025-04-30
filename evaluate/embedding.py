#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embedding similarity module, provides similarity scoring based on text embeddings
"""

import time
import numpy as np
from collections import Counter
from core import timed_print, HAVE_LOCAL_EMBEDDING_MODEL, LOCAL_MODEL_DIR, EMBEDDING_MODEL_NAME

def calculate_embedding_score(hypothesis, reference, silent=False):
    """
    Calculate text embedding similarity score
    
    Args:
        hypothesis: Model response text
        reference: Reference response text
        silent: Whether to suppress detailed output
        
    Returns:
        Embedding similarity score, range 0-1
    """
    try:
        # Check if input is empty
        if not hypothesis or not reference:
            if not silent:
                timed_print("Warning: Embedding similarity calculation received empty input")
            return 0.0
        
        # Ensure input is string
        hypothesis = str(hypothesis)
        reference = str(reference)
        
        # Record text length for monitoring
        hyp_len = len(hypothesis)
        ref_len = len(reference)
        
        # Initialize embedding model
        start_time = time.time()
        try:
            from sentence_transformers import SentenceTransformer, util
            
            # Use local model or small multilingual model
            if HAVE_LOCAL_EMBEDDING_MODEL:
                model_path = f"{LOCAL_MODEL_DIR}/{EMBEDDING_MODEL_NAME}"
                if not silent:
                    timed_print(f"Using local model: {model_path}")
                model_name = model_path
            else:
                # Use small multilingual model as fallback
                model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
                if not silent:
                    timed_print(f"Using remote model: {model_name}")
            
            # Check if model is cached
            embedding_models = {}
            if model_name not in embedding_models:
                load_start = time.time()
                if not silent:
                    timed_print(f"Loading embedding model: {model_name}")
                embedding_models[model_name] = SentenceTransformer(model_name)
                load_time = time.time() - load_start
                if load_time > 2.0 and not silent:
                    timed_print(f"Embedding model loading time: {load_time:.2f} seconds")
            
            model = embedding_models[model_name]
            
            # Encode text to get embeddings
            encode_start = time.time()
            hyp_embedding = model.encode(hypothesis, convert_to_tensor=True)
            ref_embedding = model.encode(reference, convert_to_tensor=True)
            encode_time = time.time() - encode_start
            
            if encode_time > 1.0 and not silent:
                timed_print(f"Embedding encoding time: {encode_time:.2f} seconds - Input length: {hyp_len}/{ref_len} chars")
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(hyp_embedding, ref_embedding).item()
            
            total_time = time.time() - start_time
            if total_time > 1.5 and not silent:
                timed_print(f"Total embedding similarity calculation time: {total_time:.2f} seconds")
            
            return similarity
        except ImportError:
            if not silent:
                timed_print("Warning: sentence_transformers library not available, using fallback method")
            
            # Fallback method: Use simple bag-of-words model and cosine similarity
            # Tokenize Chinese text
            try:
                import jieba
                hyp_tokens = list(jieba.cut(hypothesis))
                ref_tokens = list(jieba.cut(reference))
            except ImportError:
                # If jieba is not available, simple space-based tokenization
                hyp_tokens = hypothesis.split()
                ref_tokens = reference.split()
            
            # Create vocabulary
            vocab = list(set(hyp_tokens + ref_tokens))
            
            # Create bag-of-words vectors
            def create_vector(tokens):
                counter = Counter(tokens)
                return np.array([counter.get(word, 0) for word in vocab])
            
            hyp_vec = create_vector(hyp_tokens)
            ref_vec = create_vector(ref_tokens)
            
            # Calculate cosine similarity
            norm_hyp = np.linalg.norm(hyp_vec)
            norm_ref = np.linalg.norm(ref_vec)
            
            if norm_hyp == 0 or norm_ref == 0:
                return 0.0
            
            cosine_sim = np.dot(hyp_vec, ref_vec) / (norm_hyp * norm_ref)
            
            total_time = time.time() - start_time
            if total_time > 0.5 and not silent:
                timed_print(f"Fallback embedding calculation time: {total_time:.2f} seconds")
            
            return cosine_sim
    except Exception as e:
        if not silent:
            timed_print(f"Error calculating embedding similarity: {str(e)}")
        return 0.0

def load_embedding_model():
    """Load embedding model"""
    try:
        from sentence_transformers import SentenceTransformer
        
        if HAVE_LOCAL_EMBEDDING_MODEL:
            model_path = f"{LOCAL_MODEL_DIR}/{EMBEDDING_MODEL_NAME}"
            timed_print(f"Loading local embedding model: {model_path}")
            model = SentenceTransformer(model_path)
        else:
            # Use small multilingual model as fallback
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            timed_print(f"Loading remote embedding model: {model_name}")
            model = SentenceTransformer(model_name)
        
        return model
    except Exception as e:
        timed_print(f"Failed to load embedding model: {e}")
        return None 