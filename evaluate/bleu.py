#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLEU scoring module, provides calculation of BLEU series metrics
Supports multiple languages (English, Chinese, Japanese)
"""

from core import timed_print
from tokenizers import tokenize_text

def calculate_bleu_scores(hypothesis, reference, silent=False):
    """
    Calculate BLEU scores
    
    Args:
        hypothesis: Model response text
        reference: Reference response text
        silent: Whether to suppress detailed output
        
    Returns:
        Dictionary containing bleu1 to bleu4 scores
    """
    # Check if input is empty
    if not hypothesis or not reference:
        if not silent:
            timed_print("Warning: BLEU score calculation received empty input")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Ensure input is string
        hypothesis = str(hypothesis)
        reference = str(reference)
        
        # Tokenization
        hyp_tokens = tokenize_text(hypothesis)
        ref_tokens = [tokenize_text(reference)]  # Reference answer needs to be a list of lists
        
        # Use smoothing function
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU scores at different levels
        bleu1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4}
    except Exception as e:
        if not silent:
            timed_print(f"Error calculating BLEU scores: {str(e)}")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

def calculate_improved_bleu_score(hypothesis, reference, silent=False):
    """
    Calculate improved BLEU score for Japanese short texts
    
    Args:
        hypothesis: Model response text
        reference: Reference response text
        silent: Whether to suppress detailed output
        
    Returns:
        Dictionary containing bleu1 to bleu4 scores
    """
    # Ensure input is string
    hypothesis = str(hypothesis)
    reference = str(reference)
    
    # Character-level tokenization
    hyp_chars = list(hypothesis)
    ref_chars = [list(reference)]
    
    if not silent:
        print("Using improved BLEU calculation method (character-based)")
        print(f"Hypothesis text character count: {len(hyp_chars)}")
        print(f"Reference text character count: {len(ref_chars[0])}")
    
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Use smoothing function
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU scores at different levels
        bleu1 = sentence_bleu(ref_chars, hyp_chars, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu(ref_chars, hyp_chars, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = sentence_bleu(ref_chars, hyp_chars, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu(ref_chars, hyp_chars, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        if not silent:
            print(f"Improved BLEU scores: BLEU-1={bleu1}, BLEU-2={bleu2}, BLEU-3={bleu3}, BLEU-4={bleu4}")
        
        return {'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4}
    except Exception as e:
        if not silent:
            print(f"Error calculating improved BLEU scores: {e}")
        
        # Return zero scores on error
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0} 