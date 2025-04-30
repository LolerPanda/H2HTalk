#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report generator, provides evaluation reports in various formats
"""

import os
import json
import csv
from core import timed_print

def save_metrics_to_csv(metrics_dict, output_dir, model_name):
    """
    Save evaluation metrics in CSV format
    
    Args:
        metrics_dict: Dictionary containing evaluation metrics
        output_dir: Output directory
        model_name: Model name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create model-specific directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save main metrics to metrics.csv
    metrics_csv_path = os.path.join(model_dir, "metrics.csv")
    
    # Ensure dictionary contains all required keys
    if "composite_score" not in metrics_dict:
        metrics_dict["composite_score"] = 0.0
    
    # Ensure model name is in dictionary
    metrics_dict["model"] = model_name
    
    # Select metrics to save and map composite score field name
    csv_data = {}
    for key, value in metrics_dict.items():
        if key == "composite_score":
            csv_data["comprehensive_score"] = value
        elif key in ["rouge1", "rouge2", "rougeL", "bertscore", "embedding_score", 
                    "bleu1", "bleu2", "bleu3", "bleu4", "model"]:
            csv_data[key] = value
    
    # Field order
    metrics_order = [
        "rouge1", "rouge2", "rougeL", 
        "bertscore", "embedding_score", 
        "bleu1", "bleu2", "bleu3", "bleu4", 
        "comprehensive_score", "model"
    ]
    
    # Write to CSV file
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = metrics_order
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_data)
    
    timed_print(f"Saved evaluation results for model {model_name} to: {metrics_csv_path}")
    
    # If there are scenario evaluation results, save separate CSV for each scenario
    if "scenarios" in metrics_dict:
        scenarios = metrics_dict["scenarios"]
        
        for scenario_name, scenario_metrics in scenarios.items():
            # Create scenario filename, ensure it's valid
            safe_scenario_name = scenario_name.replace(" ", "_").replace("/", "_").replace("/", "_")
            scenario_csv_path = os.path.join(model_dir, f"{safe_scenario_name}_metrics.csv")
            
            # Add model name
            scenario_metrics["model"] = model_name
            
            # Convert field names
            scenario_csv_data = {}
            for key, value in scenario_metrics.items():
                if key == "composite_score":
                    scenario_csv_data["comprehensive_score"] = value
                elif key in ["rouge1", "rouge2", "rougeL", "bertscore", "embedding_score", 
                           "bleu1", "bleu2", "bleu3", "bleu4", "model"]:
                    scenario_csv_data[key] = value
            
            # Write to CSV file
            with open(scenario_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = metrics_order
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(scenario_csv_data)
            
            timed_print(f"Saved evaluation results for scenario {scenario_name} to: {scenario_csv_path}")
    
    return metrics_csv_path

def save_all_metrics_to_csv(all_metrics, output_dir):
    """
    Save evaluation metrics for all models in a single CSV file
    
    Args:
        all_metrics: Dictionary containing evaluation metrics for all models
        output_dir: Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Main metrics file
    metrics_csv_path = os.path.join(output_dir, "report.csv")
    
    # Define field order
    metrics_order = [
        "comprehensive_score", "model"
    ]
    
    # Prepare CSV data
    csv_data = []
    for model_name, metrics_dict in all_metrics.items():
        # Rename composite score key to match expected format
        row = {
            "comprehensive_score": metrics_dict.get("composite_score", 0.0),
            "model": model_name
        }
        csv_data.append(row)
    
    # Sort by comprehensive score
    csv_data.sort(key=lambda x: x["comprehensive_score"], reverse=True)
    
    # Write to CSV file
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = metrics_order
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in csv_data:
            writer.writerow(row)
    
    timed_print(f"Saved comprehensive evaluation results for all models to: {metrics_csv_path}")
    
    return metrics_csv_path 