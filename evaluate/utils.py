#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions module, providing various helper functions
"""
import os
from core import timed_print

def calculate_comprehensive_score(metrics):
    """
    Calculate comprehensive score based on weighted calculation of various metrics.
    
    Args:
        metrics: Dictionary containing various evaluation metrics
        
    Returns:
        Comprehensive score, ranging from 0-1
    """
    print("Calculating comprehensive score...")
    weights = {
        'rouge1': 0.20,
        'rouge2': 0.10,
        'rougeL': 0.20,
        'bertscore': 0.25,
        'embedding_score': 0.15,
        'bleu1': 0.05,
        'bleu4': 0.05
    }
    
    # Initialize comprehensive score and total valid weights
    score = 0.0
    total_weight = 0.0
    
    # Print score for each metric
    for metric, weight in weights.items():
        metric_value = metrics.get(metric, 0.0)
        if metric_value == 'N/A' or metric_value is None:
            print(f"  {metric}: N/A (weight: {weight:.2f})")
            continue
        
        try:
            metric_value = float(metric_value)
            weighted_score = metric_value * weight
            score += weighted_score
            total_weight += weight
            print(f"  {metric}: {metric_value:.4f} × {weight:.2f} = {weighted_score:.4f}")
        except (ValueError, TypeError) as e:
            print(f"  {metric}: Cannot calculate - {e} (weight: {weight:.2f})")
    
    # Return 0 if no valid metrics
    if total_weight == 0:
        print("  Warning: No valid metrics, comprehensive score is 0")
        return 0.0
    
    # Normalize score
    normalized_score = score / total_weight
    print(f"Comprehensive score: {normalized_score:.4f} (Total valid weights: {total_weight:.2f})")
    
    return normalized_score

def process_model_output(raw_data):
    """
    Process raw model output data, convert it to evaluable format
    Supports multiple formats including dialogue format and traditional format
    
    Args:
        raw_data: Raw data, could be dialogue list or dictionary format
        
    Returns:
        Processed data list, each item contains human and answer fields
    """
    results = []
    
    # Process dialogue format - list format
    if isinstance(raw_data, list):
        # First check if it's a dialogue list, each dialogue contains multiple turns
        for dialog_idx, dialog in enumerate(raw_data):
            if isinstance(dialog, dict) and "messages" in dialog:
                messages = dialog["messages"]
                
                # Extract all human-model dialogue turns
                for i in range(len(messages) - 1):
                    if i + 1 < len(messages) and messages[i].get("from") == "human" and messages[i+1].get("from") in ["gpt", "assistant"]:
                        result = {}
                        result["human"] = messages[i].get("value", "")
                        result["answer"] = messages[i+1].get("value", "")
                        
                        # Check if reference answer exists (could be in next position or in config)
                        if i + 2 < len(messages) and messages[i+2].get("from") == "data":
                            result["data"] = messages[i+2].get("value", "")
                        
                        # Only add if both question and answer exist
                        if result["human"] and result["answer"]:
                            results.append(result)
    
    # Process traditional format - dictionary format
    elif isinstance(raw_data, dict):
        if 'results' in raw_data:
            raw_results = raw_data['results']
            if isinstance(raw_results, list):
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                        
                    result = {}
                    # Check multiple key formats
                    human = item.get('human') or item.get('question') or item.get('input') or item.get('prompt')
                    answer = item.get('answer') or item.get('gpt') or item.get('output') or item.get('response')
                    reference = item.get('data') or item.get('expected') or item.get('gold')
                    
                    if human:
                        result["human"] = human
                    if answer:
                        result["answer"] = answer 
                    if reference:
                        result["data"] = reference
                    
                    # Only add dialogues with both question and reply
                    if "human" in result and "answer" in result:
                        results.append(result)
        # Directly a list containing messages
        elif 'messages' in raw_data:
            messages = raw_data['messages']
            for i in range(len(messages) - 1):
                if i + 1 < len(messages) and messages[i].get("from") == "human" and messages[i+1].get("from") in ["gpt", "assistant"]:
                    result = {}
                    result["human"] = messages[i].get("value", "")
                    result["answer"] = messages[i+1].get("value", "")
                    
                    # Check reference answer
                    if i + 2 < len(messages) and messages[i+2].get("from") == "data":
                        result["data"] = messages[i+2].get("value", "")
                    
                    if result["human"] and result["answer"]:
                        results.append(result)
    
    timed_print(f"Extracted {len(results)} Q&A pairs from raw data")
    return results

def merge_evaluation_reports(reports_dir, output_dir):
    """
    Merge multiple evaluation reports
    
    Args:
        reports_dir: Directory containing multiple evaluation reports
        output_dir: Output directory
    """
    import os
    import pandas as pd
    
    print(f"Merging evaluation reports: {reports_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all reports
    all_metrics = {}
    model_dirs = [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d)) and d != 'logs' and d != 'summary']
    
    if not model_dirs:
        print("Error: No model evaluation directories found")
        return
    
    print(f"Found {len(model_dirs)} model evaluation directories")
    
    # Load all reports
    for model_dir in model_dirs:
        model_name = model_dir
        csv_file = os.path.join(reports_dir, model_dir, 'metrics.csv')
        
        if os.path.exists(csv_file):
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Extract metrics
                    metrics = df.iloc[0].to_dict()
                    # Remove unnecessary columns
                    if 'model' in metrics:
                        del metrics['model']
                    
                    all_metrics[model_name] = metrics
                    print(f"  Loaded evaluation metrics for model {model_name}")
            except Exception as e:
                print(f"  Failed to load evaluation metrics for model {model_name}: {e}")
        else:
            print(f"  Evaluation metrics file not found for model {model_name}: {csv_file}")
    
    if not all_metrics:
        print("Error: No evaluation metrics loaded")
        return
    
    # Call visualization module functions to generate merged report
    try:
        from visualizations import generate_report, generate_visualizations
        # Generate merged report
        generate_report(all_metrics, output_dir)
        
        # Generate visualization charts
        generate_visualizations(all_metrics, output_dir)
        
        print(f"Merged report generated to: {output_dir}")
    except Exception as e:
        print(f"Error generating merged report: {e}")
    
    return all_metrics 

# Path handling utility functions
def ensure_path_exists(path):
    """Ensure path exists, create if it doesn't"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def normalize_path(path):
    """Normalize path, ensure using current system's path separator"""
    return os.path.normpath(path)

def safe_join_path(*paths):
    """Safely join paths, handle path separators for different systems"""
    return os.path.normpath(os.path.join(*paths))
