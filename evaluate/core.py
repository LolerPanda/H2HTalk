#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core functionality module, containing basic configuration and common utility functions
"""

import os
import sys
import io
import time
from datetime import datetime

# Redirect standard output and error streams to ensure Unicode support
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(current_dir, "evaluation_results")
OUTPUT_DIR = RESULTS_DIR
MODEL_RESULTS_DIR = os.path.join(current_dir, "formatted_results")
REF_RESULTS_DIR = os.path.join(current_dir, "processed_data", "json_output")
OUTPUT_PREFIX = "model_metrics"
LOCAL_MODEL_DIR = os.path.join(current_dir, "models")

# Model configuration
BERTSCORE_MODEL_NAME = "bert-base-chinese_sentence_transformer"
EMBEDDING_MODEL_NAME = "m3e-base_sentence_transformer"

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add timestamp output function
def timed_print(message, include_memory=False):
    """Output message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    memory_info = ""
    if include_memory and hasattr(sys, 'getrefcount'):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            memory_info = f" [Memory: {memory_mb:.1f}MB]"
        except:
            pass
    print(f"[{timestamp}]{memory_info} {message}")

def get_memory_usage():
    """
    Get current process memory usage (MB)
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except:
        # If unable to get memory usage, return estimated value
        return 500.0  # Default 500MB

# Check if local models are available
def check_local_models():
    """
    Check if local models are downloaded
    
    Returns:
        A dictionary containing local model names and paths
    """
    timed_print("Checking local models...")
    local_models = {}
    bertscore_model_path = os.path.join(LOCAL_MODEL_DIR, BERTSCORE_MODEL_NAME)
    embedding_model_path = os.path.join(LOCAL_MODEL_DIR, EMBEDDING_MODEL_NAME)
    
    bertscore_available = os.path.exists(bertscore_model_path)
    embedding_available = os.path.exists(embedding_model_path)
    
    if bertscore_available:
        local_models["BERTScore Model"] = bertscore_model_path
    else:
        timed_print(f"Warning: BERTScore local model does not exist: {bertscore_model_path}")
        timed_print("Please run python download_models.py to download the model first")
    
    if embedding_available:
        local_models["Embedding Model"] = embedding_model_path
    else:
        timed_print(f"Warning: Embedding local model does not exist: {embedding_model_path}")
        timed_print("Please run python download_models.py to download the model first")
    
    return local_models

# Check models during module initialization
timed_print("Starting to check local models...")
local_models = check_local_models()
HAVE_LOCAL_BERTSCORE_MODEL = "BERTScore Model" in local_models
HAVE_LOCAL_EMBEDDING_MODEL = "Embedding Model" in local_models

if HAVE_LOCAL_BERTSCORE_MODEL:
    timed_print(f"Detected local BERTScore model: {local_models['BERTScore Model']}")
if HAVE_LOCAL_EMBEDDING_MODEL:
    timed_print(f"Detected local Embedding model: {local_models['Embedding Model']}")

# Create log directory
log_dir = os.path.join(current_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timed_print(f"Created log directory: {log_dir}")

# Initialize library availability check
try:
    import jieba
    HAVE_JIEBA = True
    # Pre-initialize jieba
    jieba.initialize()
    timed_print("Successfully imported and initialized jieba library")
except ImportError:
    HAVE_JIEBA = False
    timed_print("Warning: jieba library not installed, will affect Chinese word segmentation")

try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    HAVE_PYCOCOEVALCAP = True
    timed_print("Successfully imported pycocoevalcap library")
except ImportError:
    HAVE_PYCOCOEVALCAP = False
    timed_print("pycocoevalcap library not found, will use built-in evaluation methods")

try:
    import bert_score
    HAVE_BERT_SCORE = True
    timed_print("Successfully imported bert-score library")
    # Use bert_score to avoid "not accessed" warning
    _ = bert_score.__version__
except ImportError:
    HAVE_BERT_SCORE = False
    timed_print("bert-score library not found, will use built-in evaluation methods")

try:
    import fugashi
    HAVE_FUGASHI = True
    timed_print("Successfully imported fugashi library")
except ImportError:
    HAVE_FUGASHI = False
    timed_print("Warning: fugashi library not installed, will affect Japanese word segmentation") 