#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluator module, provides functionality for evaluating model responses
"""

import os
import time
import concurrent.futures
from datetime import datetime
from core import timed_print, get_memory_usage
from rouge import calculate_rouge_scores
from bleu import calculate_bleu_scores
from embedding import calculate_embedding_score
from bertscore import calculate_bertscore

def evaluate_text_metrics(hypothesis, reference):
    """
    Comprehensive text metrics evaluation
    
    Args:
        hypothesis: Model generated text
        reference: Reference text
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    if not hypothesis or not reference:
        return {
            'rouge1': 0.0, 
            'rouge2': 0.0, 
            'rougeL': 0.0, 
            'bleu1': 0.0,
            'bleu2': 0.0,
            'bleu3': 0.0,
            'bleu4': 0.0,
            'bertscore': 0.0,
            'embedding_score': 0.0  # Ensure initial value exists
        }
    
    # Get language type (currently test data is in Chinese)
    lang = 'zh'  # Force using Chinese as evaluation language
    # Silent mode, no debug output
    silent = False
    
    # Dictionary to store various metrics
    metrics = {}
    
    # If input is valid, calculate metrics
    if hypothesis and reference:
        # Calculate ROUGE scores
        rouge_scores = calculate_rouge_scores(hypothesis, reference, silent)
        if isinstance(rouge_scores, dict):
            # If return is dictionary, update directly
            metrics.update(rouge_scores)
        else:
            # Compatible with old version return value (tuple)
            try:
                if len(rouge_scores) == 3:
                    metrics['rouge1'] = rouge_scores[0]
                    metrics['rouge2'] = rouge_scores[1]
                    metrics['rougeL'] = rouge_scores[2]
            except:
                # If error occurs, use default values
                timed_print('rouge calculate error')
                metrics['rouge1'] = 0.0
                metrics['rouge2'] = 0.0
                metrics['rougeL'] = 0.0
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu_scores(hypothesis, reference, silent)
        if isinstance(bleu_scores, dict):
            metrics.update(bleu_scores)
        else:
            # If return is not dictionary, use default values
            metrics['bleu1'] = 0.0
            metrics['bleu2'] = 0.0
            metrics['bleu3'] = 0.0
            metrics['bleu4'] = 0.0
        
        # Calculate BERTScore
        bert_score = calculate_bertscore(hypothesis, reference, silent)
        # Handle case where return value might be a list
        if isinstance(bert_score, list):
            metrics['bertscore'] = bert_score[0] if bert_score else 0.0
        else:
            metrics['bertscore'] = bert_score
        
        # Calculate Embedding Score (text embedding similarity)
        embedding_score = calculate_embedding_score(hypothesis, reference, silent)
        # Handle case where return value might be a list
        if isinstance(embedding_score, list):
            metrics['embedding_score'] = embedding_score[0] if embedding_score else 0.0
        else:
            metrics['embedding_score'] = embedding_score
    
    # Ensure all necessary metrics exist
    required_metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'bertscore', 'embedding_score']
    for metric in required_metrics:
        if metric not in metrics:
            metrics[metric] = 0.0
    
    return metrics

def evaluate_response(model_response, reference_response):
    """
    Evaluate various metrics between model response and data response
    
    Args:
        model_response: Model's response text
        reference_response: Reference response text
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    metrics = {}
    
    # Calculate various metrics
    start_time = time.time()
    rouge_scores = calculate_rouge_scores(model_response, reference_response)
    rouge_time = time.time() - start_time
    
    start_time = time.time()
    bleu_scores = calculate_bleu_scores(model_response, reference_response)
    bleu_time = time.time() - start_time
    
    start_time = time.time()
    bertscore = calculate_bertscore(model_response, reference_response)
    bert_time = time.time() - start_time
    
    start_time = time.time()
    embedding_score = calculate_embedding_score(model_response, reference_response)
    embed_time = time.time() - start_time
    
    # If any calculation takes more than 0.5 seconds, log details
    if rouge_time > 0.5:
        timed_print(f"ROUGE calculation time: {rouge_time:.2f} seconds (slow)")
    if bleu_time > 0.5:
        timed_print(f"BLEU calculation time: {bleu_time:.2f} seconds (slow)")
    if bert_time > 0.5:
        timed_print(f"BERTScore calculation time: {bert_time:.2f} seconds (slow)")
    if embed_time > 0.5:
        timed_print(f"Embedding similarity calculation time: {embed_time:.2f} seconds (slow)")
    
    # Process ROUGE scores, which might be dictionary or tuple
    if isinstance(rouge_scores, dict):
        metrics["rouge1"] = rouge_scores["rouge1"] 
        metrics["rouge2"] = rouge_scores["rouge2"]
        metrics["rougeL"] = rouge_scores["rougeL"]
    elif isinstance(rouge_scores, tuple) and len(rouge_scores) == 3:
        metrics["rouge1"] = rouge_scores[0]
        metrics["rouge2"] = rouge_scores[1]
        metrics["rougeL"] = rouge_scores[2]
    
    # Add other metrics
    metrics["bertscore"] = bertscore
    metrics["embedding_score"] = embedding_score
    metrics["bleu1"] = bleu_scores["bleu1"]
    metrics["bleu2"] = bleu_scores["bleu2"]
    metrics["bleu3"] = bleu_scores["bleu3"]
    metrics["bleu4"] = bleu_scores["bleu4"]
    
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Add log output, record score for each message
    with open(os.path.join(log_dir, "message_evaluation_log.txt"), "a", encoding="utf-8") as log_file:
        log_file.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}/n")
        log_file.write(f"Rouge-1: {metrics['rouge1']:.4f}, Rouge-2: {metrics['rouge2']:.4f}, Rouge-L: {metrics['rougeL']:.4f}, Time: {rouge_time:.2f} seconds/n")
        log_file.write(f"BERTScore: {metrics['bertscore']:.4f}, Time: {bert_time:.2f} seconds/n")
        log_file.write(f"Embedding: {metrics['embedding_score']:.4f}, Time: {embed_time:.2f} seconds/n")
        log_file.write(f"BLEU-1: {metrics['bleu1']:.4f}, BLEU-2: {metrics['bleu2']:.4f}, BLEU-3: {metrics['bleu3']:.4f}, BLEU-4: {metrics['bleu4']:.4f}, Time: {bleu_time:.2f} seconds/n")
        log_file.write("="*50 + "/n")
    
    return metrics

def process_qa_pair(index, model_response, ref_response):
    """
    Process evaluation for a single Q&A pair
    
    Args:
        index: Q&A pair index
        model_response: Model response
        ref_response: Reference response
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        start_time = time.time()
        
        # Clean and validate input
        if not isinstance(model_response, str) or not model_response.strip():
            timed_print(f"Warning: When processing index {index}, model response is empty or not a string")
            return None
            
        if not isinstance(ref_response, str) or not ref_response.strip():
            timed_print(f"Warning: When processing index {index}, data response is empty or not a string")
            return None
        
        model_response = model_response.strip()
        ref_response = ref_response.strip()
        
        # Calculate metrics
        metrics = evaluate_text_metrics(model_response, ref_response)
        
        # Add processing time
        process_time = time.time() - start_time
        if process_time > 1.0:
            timed_print(f"Q&A pair {index} processing time is longer: {process_time:.2f} seconds")
        
        # Record detailed evaluation results to log
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "detailed_evaluation.log")
        
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"===== Question ID: {index} =====/n")
            log_file.write(f"Model response: {model_response[:100]}.../n")
            log_file.write(f"Reference response: {ref_response[:100]}.../n")
            log_file.write(f"Evaluation metrics:/n")
            log_file.write(f"  Rouge-1: {metrics['rouge1']:.4f}/n")
            log_file.write(f"  Rouge-2: {metrics['rouge2']:.4f}/n")
            log_file.write(f"  Rouge-L: {metrics['rougeL']:.4f}/n")
            log_file.write(f"  BERTScore: {metrics['bertscore']:.4f}/n")
            log_file.write(f"  Embedding Score: {metrics['embedding_score']:.4f}/n")
            log_file.write(f"  BLEU-1: {metrics['bleu1']:.4f}/n")
            log_file.write(f"  BLEU-2: {metrics['bleu2']:.4f}/n")
            log_file.write(f"  BLEU-3: {metrics['bleu3']:.4f}/n")
            log_file.write(f"  BLEU-4: {metrics['bleu4']:.4f}/n")
            log_file.write(f"Processing time: {process_time:.2f} seconds/n")
            log_file.write("="*50 + "/n\n")
            
        return metrics
    except Exception as e:
        timed_print(f"Error: Error when processing Q&A pair {index}: {str(e)}")
        return None

def evaluate_model(model_name, model_dir, reference_data, workers=4, sample_size=1):
    """
    Evaluate performance of a single model, only select specified number of samples for each scenario
    
    Args:
        model_name: Model name
        model_dir: Model data directory
        reference_data: Reference data
        workers: Number of worker threads
        sample_size: Number of samples to select for each scenario
    
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Record start time and memory usage
    start_time = time.time()
    timed_print(f"[Memory: {get_memory_usage():.1f}MB] Starting to evaluate model {model_name}...")
    timed_print(f"Sampling method: Select {sample_size} Q&A pairs for each scenario")
    
    # Adjust worker number to avoid resource shortage
    available_cpus = os.cpu_count() or 4
    memory_mb = get_memory_usage()
    
    # Dynamically adjust worker number
    if memory_mb > 4000:  # Memory usage exceeds 4GB
        adjusted_workers = min(workers, max(1, available_cpus // 2))
        if adjusted_workers < workers:
            timed_print(f"High memory usage, adjust worker number: {workers} -> {adjusted_workers}")
            workers = adjusted_workers
    
    timed_print(f"Using {workers} worker threads")
    
    # Load model data
    from data_loaders import load_model_data
    model_data = load_model_data(model_dir)
    if not model_data:
        timed_print(f"Error: Unable to load data for model {model_name}, skip evaluation")
        return None
    
    # Record time
    load_time = time.time() - start_time
    timed_print(f"Model data loading time: {load_time:.2f} seconds")
    
    # Perform evaluation
    all_metrics = {}
    
    # Overall evaluation
    total_scores = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bertscore": 0.0,
        "embedding_score": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0
    }
    
    # Record evaluated Q&A pairs
    total_qa_pairs = 0
    processed_qa_pairs = 0
    
    # Get all test types (i.e., scenarios, such as 'chat', 'routine', etc.)
    scenarios = list(reference_data.keys())
    timed_print(f"Evaluation scenarios: {', '.join(scenarios)}")
    
    # Save evaluation results for each scenario
    scenario_results = {}
    
    # Create worker thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        
        # Record start time
        eval_start = time.time()
        
        # Evaluate by scenario
        for scenario in scenarios:
            scenario_start = time.time()
            if scenario not in reference_data:
                timed_print(f"Warning: Reference answer not found for {scenario}, skip")
                continue
        
            ref_data = reference_data[scenario]
            if scenario not in model_data:
                timed_print(f"Warning: Model {model_name} does not have results for {scenario}, skip")
                continue
            
            # Get Q&A pairs
            qa_pairs = []
            for qa_id, ref_value in ref_data.items():
                if qa_id in model_data[scenario]:
                    model_response = model_data[scenario][qa_id]
                    ref_response = ""
                    
                    # Process different format data data
                    if isinstance(ref_value, dict):
                        # If dictionary format, try to get answer field
                        ref_response = ref_value.get("answer", "")
                    elif isinstance(ref_value, str):
                        # If string format, use directly
                        ref_response = ref_value
                    
                    if model_response and ref_response:
                        qa_pairs.append((qa_id, model_response, ref_response))
                        
                        # If sample size is reached, don't add more Q&A pairs
                        if len(qa_pairs) >= sample_size:
                            break
        
            total_qa_pairs += len(qa_pairs)
        
            timed_print(f"Scenario {scenario}: Select {len(qa_pairs)} Q&A pairs as samples (Total data Q&A pairs: {len(ref_data)})")
            
            # Submit evaluation tasks to thread pool
            for qa_id, model_response, ref_response in qa_pairs:
                future = executor.submit(process_qa_pair, qa_id, model_response, ref_response)
                futures.append((future, scenario, qa_id))
            
            scenario_time = time.time() - scenario_start
            if scenario_time > 1.0:
                timed_print(f"Setting scenario {scenario} evaluation task time: {scenario_time:.2f} seconds")
        
        # Collect results
        scenario_qa_counts = {}  # Number of Q&A pairs processed for each scenario
        scenario_scores = {}     # Total score for each scenario
        
        for future, scenario, qa_id in futures:
            try:
                metrics = future.result()
                if metrics:
                    processed_qa_pairs += 1
                    
                    # Initialize scenario count and score
                    if scenario not in scenario_qa_counts:
                        scenario_qa_counts[scenario] = 0
                        scenario_scores[scenario] = {k: 0.0 for k in total_scores.keys()}
                    
                    # Update scenario count
                    scenario_qa_counts[scenario] += 1
                    
                    # Accumulate scenario score
                    for key in total_scores:
                        if key in metrics:
                            scenario_scores[scenario][key] += metrics[key]
                    
                    # Accumulate various metrics
                    for key in total_scores:
                        if key in metrics:
                            total_scores[key] += metrics[key]
                    
                    # Record detailed Q&A pair information
                    timed_print(f"Evaluated Q&A pair {qa_id} (scenario: {scenario})")
                    for metric_name, metric_value in metrics.items():
                        timed_print(f"  {metric_name}: {metric_value:.4f}")
            except Exception as e:
                timed_print(f"Error when processing Q&A pair {qa_id}: {str(e)}")
    
    # Calculate average score for each scenario
    for scenario, count in scenario_qa_counts.items():
        if count > 0:
            scenario_results[scenario] = {}
            for key in total_scores:
                scenario_results[scenario][key] = scenario_scores[scenario][key] / count
                
            # Calculate composite score for scenario
            weights = {
                "rouge1": 0.10,
                "rouge2": 0.15,
                "rougeL": 0.20,
                "bertscore": 0.25,
                "embedding_score": 0.20,
                "bleu1": 0.03,
                "bleu2": 0.03,
                "bleu3": 0.02,
                "bleu4": 0.02
            }
            
            composite_score = 0.0
            for key, weight in weights.items():
                composite_score += scenario_results[scenario][key] * weight
                
            scenario_results[scenario]["composite_score"] = composite_score
    
    # Write scenario evaluation results to log
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{model_name}_scenario_evaluation.log")
    
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Model {model_name} Scenario Evaluation Results (Sampling method: Select {sample_size} Q&A pairs for each scenario)/n")
        log_file.write("=" * 60 + "/n\n")
        
        for scenario, metrics in scenario_results.items():
            log_file.write(f"Scenario: {scenario}/n")
            log_file.write(f"Q&A pair number: {scenario_qa_counts[scenario]}/n")
            log_file.write(f"Composite score: {metrics.get('composite_score', 0.0):.4f}/n")
            log_file.write("Various metrics:/n")
            
            for key in ["rouge1", "rouge2", "rougeL", "bertscore", "embedding_score", 
                       "bleu1", "bleu2", "bleu3", "bleu4"]:
                log_file.write(f"  {key}: {metrics.get(key, 0.0):.4f}/n")
                
            log_file.write("/n" + "-" * 40 + "/n\n")
    
    timed_print(f"Saved scenario evaluation results to: {log_file_path}")
    
    # Calculate average score
    if processed_qa_pairs > 0:
        for key in total_scores:
            total_scores[key] /= processed_qa_pairs
    
    # Calculate composite score
    # Weights for various metrics
    weights = {
        "rouge1": 0.10,
        "rouge2": 0.15,
        "rougeL": 0.20,
        "bertscore": 0.25,
        "embedding_score": 0.20,
        "bleu1": 0.03,
        "bleu2": 0.03,
        "bleu3": 0.02,
        "bleu4": 0.02
    }
    
    # Calculate weighted average score
    composite_score = 0.0
    for key, weight in weights.items():
        composite_score += total_scores[key] * weight
    
    # Add to results
    all_metrics = total_scores.copy()
    all_metrics["composite_score"] = composite_score
    all_metrics["scenarios"] = scenario_results  # Add scenario evaluation results
    
    # End timing
    total_eval_time = time.time() - start_time
    eval_only_time = time.time() - eval_start
    
    # Output evaluation results
    timed_print(f"Evaluation completed! Processed {processed_qa_pairs}/{total_qa_pairs} Q&A pairs")
    timed_print(f"Evaluation time: {eval_only_time:.2f} seconds, Total time: {total_eval_time:.2f} seconds")
    timed_print(f"Composite score: {composite_score:.4f}")
    
    for key, value in total_scores.items():
        timed_print(f"{key}: {value:.4f}")
    
    return all_metrics 