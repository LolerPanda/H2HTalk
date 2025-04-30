#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Companion Model Evaluation Tool - Modular Version
Evaluate companion models using ROUGE, BERTScore and BLEU metrics
"""

import os
import json
import argparse
import time
from core import timed_print, get_memory_usage, MODEL_RESULTS_DIR, RESULTS_DIR, REF_RESULTS_DIR
from data_loaders import load_model_data, load_reference_data
from evaluator import evaluate_model
from visualizations import generate_visualizations, generate_report, generate_html_report

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate model output similarity with data answers")
    
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Reference answer directory")
    parser.add_argument("--formatted-results-dir", type=str, default=None,
                        help="Formatted model output directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name to evaluate")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processing workers")
    parser.add_argument("--sample-size", type=int, default=1,
                        help="Number of samples to evaluate per scenario")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Output basic information
    timed_print(f"[Memory: {get_memory_usage():.1f}MB] Starting evaluation...")
    timed_print(f"Output directory: {output_dir}")
    
    # Load data data
    reference_dir = args.reference_dir or REF_RESULTS_DIR
    timed_print(f"Loading data data: {reference_dir}")
    reference_data = load_reference_data(reference_dir)
    
    # If model name is specified, only evaluate that model
    model_names = []
    if args.model:
        model_names = [args.model]
        timed_print(f"Evaluating specific model: {args.model}")
    else:
        # Otherwise evaluate all available models
        model_dir = args.formatted_results_dir or MODEL_RESULTS_DIR
        available_models = get_available_models(model_dir)
        model_names = available_models
        timed_print(f"Found {len(model_names)} model directories")
    
    # Sample size
    sample_size = args.sample_size
    timed_print(f"Will evaluate {sample_size} Q&A pairs per scenario")
    
    # Evaluate all models
    all_model_metrics = {}
    
    for model_name in model_names:
        timed_print(f"/nStarting evaluation for model: {model_name}")
        
        model_dir = os.path.join(args.formatted_results_dir or MODEL_RESULTS_DIR, model_name)
        metrics = evaluate_model(model_name, model_dir, reference_data, workers=args.workers, sample_size=sample_size)
        
        if metrics:
            all_model_metrics[model_name] = metrics
            
            # Save evaluation results for single model using new function
            save_model_evaluation(model_name, metrics, output_dir)
    
    # Save evaluation results for all models using new function
    save_all_evaluations(all_model_metrics, output_dir)
    
    # Generate detailed report
    timed_print("Generating detailed report...")
    
    # Import report generator (using existing generate_report function)
    try:
        report_path = generate_report(all_model_metrics, output_dir)
        timed_print(f"Evaluation report generated: {report_path}")
    except Exception as e:
        timed_print(f"Error generating report: {str(e)}")
    
    # Generate visualizations
    timed_print("Generating visualizations...")
    
    # Import visualization module
    try:
        viz_paths = generate_visualizations(all_model_metrics, output_dir)
        for viz_type, viz_path in viz_paths.items():
            timed_print(f"{viz_type} generated: {viz_path}")
    except Exception as e:
        timed_print(f"Error generating visualizations: {str(e)}")
    
    # Generate HTML report
    timed_print("Generating HTML report...")
    
    # Import HTML report generator
    try:
        html_path = generate_html_report(all_model_metrics, output_dir)
        timed_print(f"HTML report generated: {html_path}")
    except Exception as e:
        timed_print(f"Error generating HTML report: {str(e)}")
    
    # Calculate total time
    total_time = time.time() - start_time
    timed_print(f"/nEvaluation completed! Total time: {total_time:.2f} seconds")
    timed_print(f"Results saved to: {output_dir}")
    
    return all_model_metrics

def save_model_evaluation(model_name, metrics, output_dir):
    """
    Save model evaluation results
    
    Args:
        model_name: Model name
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Use report_generator module's save_metrics_to_csv function
    from report_generator import save_metrics_to_csv
    metrics_csv_path = save_metrics_to_csv(metrics, output_dir, model_name)
    
    # Save detailed results in JSON format
    json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    timed_print(f"Saved evaluation results for model {model_name} to: {json_path}")
    
    return metrics_csv_path, json_path

def save_all_evaluations(all_metrics, output_dir):
    """
    Save evaluation results for all models
    
    Args:
        all_metrics: Evaluation metrics for all models
        output_dir: Output directory
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Use report_generator module's save_all_metrics_to_csv function
    from report_generator import save_all_metrics_to_csv
    metrics_csv_path = save_all_metrics_to_csv(all_metrics, output_dir)
    
    # Save detailed results in JSON format
    json_path = os.path.join(output_dir, "all_models_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    timed_print(f"Saved evaluation results for all models to: {json_path}")
    
    return metrics_csv_path, json_path

def get_available_models(model_dir):
    """
    Get list of available models
    
    Args:
        model_dir: Model data directory
        
    Returns:
        List of model names
    """
    if not os.path.exists(model_dir):
        timed_print(f"Warning: Model directory does not exist: {model_dir}")
        return []
    
    # Get all subdirectories (each subdirectory represents a model)
    models = [d for d in os.listdir(model_dir) 
              if os.path.isdir(os.path.join(model_dir, d))]
    
    return models

if __name__ == "__main__":
    main() 