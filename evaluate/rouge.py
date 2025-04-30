#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROUGE scoring module, provides calculation of ROUGE series metrics
Supports multiple languages (English, Chinese, Japanese)
Added support for pycocoevalcap library
"""

import time
import random
from core import timed_print, HAVE_PYCOCOEVALCAP
from tokenizers import tokenize_text

def calculate_rouge_scores(hypothesis, reference, silent=False):
    """
    Calculate ROUGE scores
    
    Args:
        hypothesis: Model response text
        reference: Reference response text
        silent: Whether to suppress detailed output
        
    Returns:
        Dictionary containing rouge1 to rougeL scores
    """
    # Check if input is empty
    if not hypothesis or not reference:
        if not silent:
            timed_print("Warning: ROUGE score calculation received empty input")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return calculate_custom_rouge(hypothesis, reference, True, silent)
    try:
        # # First try using pycocoevalcap library (if available)
        # if HAVE_PYCOCOEVALCAP:
        #     pycoco_result = calculate_pycoco_rouge(hypothesis, reference, silent)
            
        #     if pycoco_result is not None:
        #         if not silent:
        #             timed_print("Successfully used coco to calculate rouge")
        #         return pycoco_result
        
        # If pycocoevalcap is not available or fails, fall back to rouge-score library
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        # Calculate ROUGE scores
        start = time.time()
        scores = scorer.score(hypothesis, reference)
        calc_time = time.time() - start
        
        # If calculation time is abnormal, record details for debugging
        if calc_time > 0.5 and not silent:
            hyp_tokens = hypothesis.split()
            ref_tokens = reference.split()
            timed_print(f"ROUGE calculation took longer than expected ({calc_time:.2f} seconds) - Input length: {len(hyp_tokens)} words/{len(hypothesis)} chars vs {len(ref_tokens)} words/{len(reference)} chars")
        
        result = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
        return result
    except Exception as e:
        if not silent:
            timed_print(f"Error calculating ROUGE scores: {str(e)}")
        # Try using custom ROUGE calculation
        try:
            return calculate_custom_rouge(hypothesis, reference, True, silent)
        except Exception as e2:
            if not silent:
                timed_print(f"Custom ROUGE calculation also failed: {str(e2)}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def calculate_pycoco_rouge(hypothesis, reference, silent=False):
    """
    Calculate ROUGE scores using pycocoevalcap library
    
    Args:
        hypothesis: Model response text
        reference: Reference response text
        silent: Whether to suppress detailed output
        
    Returns:
        Dictionary containing rouge scores, returns None on failure
    """
    try:
        # Import Rouge class from pycocoevalcap
        from pycocoevalcap.rouge.rouge import Rouge
        
        # Generate a unique ID for each sample
        sample_id = f"sample_{random.randint(1000, 9999)}"
        
        # pycocoevalcap expects a specific dictionary format
        hyp = {sample_id: [hypothesis]}
        ref = {sample_id: [reference]}
        
        # Create Rouge evaluator
        start_time = time.time()
        rouge_evaluator = Rouge()
        
        # Calculate ROUGE scores
        rouge_scores = rouge_evaluator.compute_score(ref, hyp)
        calc_time = time.time() - start_time
        
        # pycocoevalcap returns a tuple of (score, score list)
        if isinstance(rouge_scores, tuple) and len(rouge_scores) >= 1:
            rouge_score = rouge_scores[0]  # Get average score
        else:
            rouge_score = rouge_scores
        
        if not silent and calc_time > 0.5:
            timed_print(f"pycocoevalcap ROUGE calculation time: {calc_time:.2f} seconds")
        
        # Note: pycocoevalcap's Rouge class only provides ROUGE-L score
        # We need to get ROUGE-1 and ROUGE-2 through other means
        # To maintain compatibility with the original function, we call custom calculation for these scores
        custom_scores = calculate_custom_rouge(hypothesis, reference, True, True)
        
        result = {
            "rouge1": custom_scores["rouge1"],
            "rouge2": custom_scores["rouge2"],
            "rougeL": rouge_score
        }
        
        if not silent:
            timed_print(f"ROUGE-L score calculated using pycocoevalcap: {rouge_score:.4f}")
        
        return result
    except Exception as e:
        if not silent:
            timed_print(f"Error calculating ROUGE with pycocoevalcap: {str(e)}")
        # Return None to indicate need to try other methods
        return None

def calculate_custom_rouge(hypothesis, reference, is_chinese=True, silent=False):
    """
    Calculate ROUGE scores using custom method
    Supports multiple languages, especially Chinese
    
    Args:
        hypothesis: Model response text
        reference: Reference response text
        is_chinese: Whether the text is in Chinese
        silent: Whether to suppress detailed output
        
    Returns:
        Dictionary containing rouge1, rouge2 and rougeL scores
    """
    try:
        # Use tokenize_text function for Chinese text segmentation
        if is_chinese:
            if not silent:
                timed_print("Using tokenize_text for Chinese text segmentation to calculate ROUGE")
            hyp_tokens = tokenize_text(hypothesis, True)
            ref_tokens = tokenize_text(reference, True)
        else:
            # Use space-based tokenization for English
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()
        
        if not silent:
            timed_print(f"Tokenized hypothesis text: {hyp_tokens[:10]}...")
            timed_print(f"Tokenized reference text: {ref_tokens[:10]}...")
        
        # Calculate ROUGE-1 and ROUGE-2 scores
        rouge1 = calculate_rouge_n(hyp_tokens, ref_tokens, 1)
        rouge2 = calculate_rouge_n(hyp_tokens, ref_tokens, 2)
        
        # Calculate ROUGE-L score
        rougeL = calculate_rouge_l(hyp_tokens, ref_tokens)
        
        if not silent:
            timed_print(f"Custom ROUGE scores: ROUGE-1={rouge1}, ROUGE-2={rouge2}, ROUGE-L={rougeL}")
        
        return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}
    except Exception as e:
        if not silent:
            timed_print(f"Error in custom ROUGE calculation: {e}")
        
        # Return zero scores on error
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def calculate_rouge_n(candidate_tokens, reference_tokens, n):
    """
    Calculate ROUGE-N score
    
    Args:
        candidate_tokens: Token list of candidate text
        reference_tokens: Token list of reference text
        n: n value for n-gram
        
    Returns:
        ROUGE-N score
    """
    # If token list is too short to form n-gram, return 0
    if len(candidate_tokens) < n or len(reference_tokens) < n:
        return 0.0
    
    # Generate n-grams
    def get_ngrams(tokens, n):
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    # Get n-grams for candidate and reference text
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    reference_ngrams = get_ngrams(reference_tokens, n)
    
    # Calculate overlapping n-grams
    overlap_count = 0
    for ngram, count in candidate_ngrams.items():
        overlap_count += min(count, reference_ngrams.get(ngram, 0))
    
    # Calculate recall
    total_reference_ngrams = sum(reference_ngrams.values())
    if total_reference_ngrams == 0:
        recall = 0.0
    else:
        recall = overlap_count / total_reference_ngrams
    
    # Calculate precision
    total_candidate_ngrams = sum(candidate_ngrams.values())
    if total_candidate_ngrams == 0:
        precision = 0.0
    else:
        precision = overlap_count / total_candidate_ngrams
    
    # Calculate F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def calculate_rouge_l(candidate_tokens, reference_tokens):
    """
    Calculate ROUGE-L score (based on longest common subsequence)
    
    Args:
        candidate_tokens: Token list of candidate text
        reference_tokens: Token list of reference text
        
    Returns:
        ROUGE-L score
    """
    # Calculate longest common subsequence (LCS)
    m, n = len(candidate_tokens), len(reference_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_tokens[i-1] == reference_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # If either sequence is empty, return 0
    if m == 0 or n == 0:
        return 0.0
    
    # Calculate recall
    recall = lcs_length / n
    
    # Calculate precision
    precision = lcs_length / m
    
    # Calculate F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1 