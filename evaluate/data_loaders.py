#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading module, provides functionality for loading model data and reference data
"""

import os
import json
import time
import glob
import concurrent.futures
from core import timed_print, get_memory_usage
from evaluator import process_qa_pair

def load_model_data(model_dir):
    """
    Load model generated data
    
    Args:
        model_dir: Model data directory
        
    Returns:
        Loaded model data, format: {scenario: {question_id: model_response}}
    """
    timed_print(f"Loading model data: {model_dir}")
    
    if not os.path.exists(model_dir):
        timed_print(f"Error: Model directory does not exist: {model_dir}")
        return {}
    
    model_data = {}
    loading_start = time.time()
    files_loaded = 0
    total_qa_pairs = 0
    
    # List all JSONL files
    jsonl_files = []
    for filename in os.listdir(model_dir):
        if filename.endswith(('.jsonl', '.json')):
            jsonl_files.append(os.path.join(model_dir, filename))
    
    if not jsonl_files:
        timed_print(f"Warning: No JSONL or JSON files found in directory {model_dir}")
        return {}
    
    # Process each JSONL file
    for jsonl_file in jsonl_files:
        file_start = time.time()
        scenario_name = os.path.splitext(os.path.basename(jsonl_file))[0]
        
        try:
            # Create scenario dictionary
            model_data[scenario_name] = {}
            qa_pairs_count = 0
            
            # Read JSONL file
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    try:
                        # Clean possible BOM markers and empty lines
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse JSON line
                        item = json.loads(line)
                        
                        # Set a default question ID (use index if multiple)
                        question_id = f"q{line_num:03d}"
                        
                        # Extract model response
                        if "response" in item:
                            # Standard response format
                            model_data[scenario_name][question_id] = item["response"]
                            qa_pairs_count += 1
                        elif "answer" in item:
                            # Alternative format
                            model_data[scenario_name][question_id] = item["answer"]
                            qa_pairs_count += 1
                        elif "messages" in item:
                            # Dialogue format
                            for i, msg in enumerate(item["messages"]):
                                if isinstance(msg, dict) and msg.get("from") in ["assistant", "gpt"]:
                                    model_data[scenario_name][question_id] = msg.get("value", "")
                                    qa_pairs_count += 1
                                    break
                    except json.JSONDecodeError as e:
                        timed_print(f"Warning: Unable to parse line {line_num}: {e}")
                    except Exception as e:
                        timed_print(f"Warning: Error processing line {line_num}: {e}")
            
            files_loaded += 1
            total_qa_pairs += qa_pairs_count
            file_time = time.time() - file_start
            
            timed_print(f"Loaded model file {os.path.basename(jsonl_file)}: {qa_pairs_count} responses (Time: {file_time:.2f} seconds)")
        
        except Exception as e:
            timed_print(f"Error processing file {jsonl_file}: {e}")
    
    loading_time = time.time() - loading_start
    timed_print(f"Model data loading completed: {files_loaded} files, {total_qa_pairs} responses, Total time: {loading_time:.2f} seconds", include_memory=True)
    
    return model_data

def load_reference_data(reference_dir):
    """
    Load reference data from reference directory
    
    Args:
        reference_dir: Reference data directory

    Returns:
        Reference data, format: {scenario_name: {question_id: reference_answer}}
    """
    timed_print(f"Loading reference answers from {reference_dir}...", include_memory=True)
    loading_start = time.time()
    
    reference_data = {}
    
    # Check if directory exists
    if not os.path.exists(reference_dir):
        timed_print(f"Error: Reference data directory does not exist: {reference_dir}")
        return {}
        
    # Get all json and jsonl files
    ref_files = []
    for ext in [".json", ".jsonl"]:
        ref_files.extend(glob.glob(os.path.join(reference_dir, f"*{ext}")))
    
    if not ref_files:
        timed_print(f"Warning: No JSON or JSONL files found in directory {reference_dir}")
        return {}
    
    timed_print(f"Found {len(ref_files)} reference files")
    
    files_loaded = 0
    total_qa_pairs = 0
    
    # Create a default scenario name for each scene file
    for file_path in ref_files:
        file_start = time.time()
        file_name = os.path.basename(file_path)
        scenario_name = os.path.splitext(file_name)[0]
        
        # Initialize scenario data dictionary
        if scenario_name not in reference_data:
            reference_data[scenario_name] = {}
        
        qa_pairs_count = 0
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            data = None
            
            for encoding in encodings:
                try:
                    # Read based on file type
                    if file_path.endswith('.jsonl'):
                        # JSONL format processing
                        with open(file_path, 'r', encoding=encoding) as f:
                            line_num = 0
                            for line in f:
                                line_num += 1
                                try:
                                    # Clean possible BOM markers and empty lines
                                    line = line.strip()
                                    if not line:
                                        continue
                                        
                                    # Parse JSON line
                                    data = json.loads(line)
                                    
                                    # Use question_id directly if it exists
                                    qa_id = data.get("question_id", f"q{line_num:03d}")
                                    
                                    # Extract answer
                                    if "response" in data:
                                        reference_data[scenario_name][qa_id] = data["response"]
                                        qa_pairs_count += 1
                                    elif "answer" in data:
                                        reference_data[scenario_name][qa_id] = data["answer"]
                                        qa_pairs_count += 1
                                    elif "data" in data:
                                        reference_data[scenario_name][qa_id] = data["data"]
                                        qa_pairs_count += 1
                                except json.JSONDecodeError as e:
                                    timed_print(f"Warning: Unable to parse line {line_num} in reference file {file_name}: {e}")
                        
                        # If successfully parsed, stop trying other encodings
                        break
                    else:
                        # JSON format processing
                        with open(file_path, 'r', encoding=encoding) as f:
                            # Try to load entire JSON file
                            data = json.load(f)
                            
                            # Process various common JSON structures
                            
                            # 1. If it's qa_pairs format
                            if isinstance(data, dict) and "qa_pairs" in data:
                                qa_pairs = data["qa_pairs"]
                                for i, qa_pair in enumerate(qa_pairs):
                                    if isinstance(qa_pair, dict):
                                        # Use provided question_id or sequence number
                                        qa_id = qa_pair.get("question_id", f"q{i+1:03d}")
                                        
                                        # Extract answer
                                        if "answer" in qa_pair:
                                            reference_data[scenario_name][qa_id] = qa_pair["answer"]
                                            qa_pairs_count += 1
                                        elif "response" in qa_pair:
                                            reference_data[scenario_name][qa_id] = qa_pair["response"]
                                            qa_pairs_count += 1
                                
                            # 2. If it's list format - possibly dialogue collection
                            elif isinstance(data, list):
                                for idx, item in enumerate(data):
                                    qa_id = f"q{idx+1:03d}"
                                    
                                    # Process dialogue format
                                    if isinstance(item, dict) and "messages" in item:
                                        messages = item["messages"]
                                        # Extract answer part from dialogue
                                        for i, msg in enumerate(messages):
                                            if isinstance(msg, dict) and msg.get("from") in ["gpt", "assistant"]:
                                                reference_data[scenario_name][qa_id] = msg.get("value", "")
                                                qa_pairs_count += 1
                                                break
                                    # Process simple QA pairs
                                    elif isinstance(item, dict) and "response" in item:
                                        reference_data[scenario_name][qa_id] = item["response"]
                                        qa_pairs_count += 1
                                    elif isinstance(item, dict) and "answer" in item:
                                        reference_data[scenario_name][qa_id] = item["answer"]
                                        qa_pairs_count += 1
                            
                            # 3. Direct key-value format {question_id: answer}
                            elif isinstance(data, dict):
                                # Single dialogue format
                                if "messages" in data:
                                    qa_id = f"q001"
                                    for msg in data["messages"]:
                                        if isinstance(msg, dict) and msg.get("from") in ["gpt", "assistant"]:
                                            reference_data[scenario_name][qa_id] = msg.get("value", "")
                                            qa_pairs_count += 1
                                            break
                                # Single response format
                                elif "response" in data:
                                    qa_id = f"q001"
                                    reference_data[scenario_name][qa_id] = data["response"]
                                    qa_pairs_count += 1
                                # Direct key-value pairs
                                else:
                                    for key, value in data.items():
                                        # Skip non-QA pair metadata fields
                                        if key in ["metadata", "info", "description", "config"]:
                                            continue
                                            
                                        if isinstance(value, dict) and "answer" in value:
                                            reference_data[scenario_name][key] = value["answer"]
                                            qa_pairs_count += 1
                                        elif isinstance(value, str):
                                            reference_data[scenario_name][key] = value
                                            qa_pairs_count += 1
                            
                            # If successfully parsed, stop trying other encodings
                            break
                except Exception as e:
                    if encoding == encodings[-1]:  # Last encoding still failed
                        timed_print(f"Error loading reference file {file_path}: {e}")
                    else:
                        continue  # Try next encoding
        except Exception as e:
            timed_print(f"Error loading reference file {file_path}: {e}")
        
        # Update statistics
        files_loaded += 1
        total_qa_pairs += qa_pairs_count
        file_time = time.time() - file_start
        
        timed_print(f"Loaded reference file {file_name}: {qa_pairs_count} QA pairs (Time: {file_time:.2f} seconds)")
    
    # Create generic reference data - ensure each model scenario has corresponding reference data
    # Create a generic reference answer locally
    reference_data["__generic__"] = {
        "q001": "This is a generic reference answer for model output evaluation.",
        "q002": "Thank you for your question, I'm happy to help you.",
        "q003": "As an AI assistant, I aim to provide accurate and useful information."
    }
    
    # If no data was loaded, use generic reference data
    if total_qa_pairs == 0:
        timed_print("No reference data loaded, will only use generic reference data for evaluation")
    
    loading_time = time.time() - loading_start
    timed_print(f"Reference data loading completed: {files_loaded} files, {total_qa_pairs} QA pairs, Total time: {loading_time:.2f} seconds", include_memory=True)
    
    return reference_data

def extract_reference_responses(scenario_data, scenario_name="", reference_data=None):
    """
    Extract QA pairs from reference data for corresponding scenario
    
    Args:
        scenario_data: Scenario data
        scenario_name: Scenario name
        reference_data: Reference data
        
    Returns:
        Extracted reference responses dictionary, format: {question: answer}
    """
    reference_responses = {}
    
    # If reference data is provided, try to extract from it
    if reference_data and scenario_name in reference_data:
        reference_responses = reference_data[scenario_name]
        print(f"  Extracted {len(reference_responses)} QA pairs from reference data")
    
    # If no reference data found, try to extract from scenario_data
    if not reference_responses and scenario_data:
        # Check if there's a results field
        if isinstance(scenario_data, dict) and 'results' in scenario_data:
            results = scenario_data['results']
            if isinstance(results, list):
                for qa in results:
                    if isinstance(qa, dict) and 'human' in qa and 'gpt' in qa:
                        reference_responses[qa['human']] = qa['gpt']
                print(f"  Extracted {len(reference_responses)} QA pairs from scenario data")
        # Check if it's messages format
        elif isinstance(scenario_data, list) and all(isinstance(item, dict) and 'messages' in item for item in scenario_data):
            for item in scenario_data:
                messages = item.get('messages', [])
                
                # Extract all human-GPT dialogue pairs
                for i in range(len(messages) - 1):
                    if i + 1 < len(messages) and messages[i].get("from") == "human" and messages[i+1].get("from") in ["gpt", "assistant"]:
                        human_msg = messages[i].get("value", "")
                        gpt_msg = messages[i+1].get("value", "")
                        
                        if human_msg and gpt_msg:
                            reference_responses[human_msg] = gpt_msg
                
            print(f"  Extracted {len(reference_responses)} QA pairs from messages format data")
    
    # If still no reference data found, use generic responses
    if not reference_responses:
        print("  No reference data found, using generic responses")
        # Create some generic QA pairs
        common_responses = {
            "Hello": "Hello, nice to meet you.",
            "Who are you": "I am an AI assistant, ready to help you.",
            "How's the weather today": "I'm not sure about the current weather, but I hope it's nice."
        }
        reference_responses.update(common_responses)
    
    return reference_responses

def extract_qa_pairs(raw_data):
    """
    Extract QA pairs from raw data, return {question: answer} dictionary
    Supports multiple formats including dialogue format and traditional format
    
    Args:
        raw_data: Raw data, could be dialogue list or dictionary format
        
    Returns:
        Extracted QA pairs dictionary, format: {question: answer}
    """
    qa_pairs = {}
    
    # Process dialogue format - list format
    if isinstance(raw_data, list):
        for dialog in raw_data:
            if isinstance(dialog, dict) and "messages" in dialog:
                messages = dialog["messages"]
                
                # Extract all QA pairs from dialogue
                for i in range(len(messages) - 1):
                    if i + 1 < len(messages):
                        # Ensure current message is human, next is GPT/reference
                        current_msg = messages[i]
                        next_msg = messages[i+1]
                        
                        if not isinstance(current_msg, dict) or not isinstance(next_msg, dict):
                            continue
                        
                        # Check if it's human-model dialogue
                        if current_msg.get("from") == "human" and (
                           next_msg.get("from") in ["gpt", "assistant", "data"]):
                            
                            human_msg = current_msg.get("value", "")
                            response_msg = next_msg.get("value", "")
                            
                            # Only add if both question and answer exist
                            if human_msg and response_msg:
                                qa_pairs[human_msg] = response_msg
    
    # Process traditional format - dictionary format
    elif isinstance(raw_data, dict):
        if 'results' in raw_data and isinstance(raw_data['results'], list):
            results = raw_data['results']
            for item in results:
                if not isinstance(item, dict):
                    continue
                    
                # Check multiple key formats
                human = item.get('human') or item.get('question') or item.get('input') or item.get('prompt')
                reference = item.get('data') or item.get('expected') or item.get('gold') or item.get('answer') or item.get('gpt')
                
                if human and reference:
                    qa_pairs[human] = reference
        
        # If it's single dialogue format
        elif 'messages' in raw_data and isinstance(raw_data['messages'], list):
            messages = raw_data['messages']
            # Extract all QA pairs from dialogue
            for i in range(len(messages) - 1):
                if i + 1 < len(messages):
                    # Ensure current message is human, next is GPT/reference
                    current_msg = messages[i]
                    next_msg = messages[i+1]
                    
                    if not isinstance(current_msg, dict) or not isinstance(next_msg, dict):
                        continue
                    
                    # Check if it's human-model dialogue
                    if current_msg.get("from") == "human" and (
                       next_msg.get("from") in ["gpt", "assistant", "data"]):
                        
                        human_msg = current_msg.get("value", "")
                        response_msg = next_msg.get("value", "")
                        
                        # Only add if both question and answer exist
                        if human_msg and response_msg:
                            qa_pairs[human_msg] = response_msg
    
    print(f"Extracted {len(qa_pairs)} QA pairs as reference data")
    return qa_pairs

def read_model_data(model_dir):
    """
    Read model data
    
    Args:
        model_dir: Model directory
        
    Returns:
        Read model data
    """
    print(f"Reading model data: {model_dir}")
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory does not exist: {model_dir}")
        return {}
    
    # Find all JSON files
    json_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        print(f"Warning: No JSON files found in directory {model_dir}")
        return {}
    
    # Read all JSON files
    model_results = {}
    for json_file in json_files:
        try:
            # Use utf-8-sig encoding to handle UTF-8 files with BOM
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                
                # Extract scenario name
                scenario = os.path.basename(json_file).replace('.json', '')
                
                # Store results
                model_results[scenario] = data
                
                print(f"Loaded data for scenario {scenario}")
        except Exception as e:
            print(f"Error reading file {json_file}: {e}")
    
    return model_results 

def evaluate_model(model_name, model_dir, reference_data, workers=4):
    """
    Evaluate performance of a single model
    
    Args:
        model_name: Model name
        model_dir: Model data directory
        reference_data: Reference data
        workers: Number of worker threads
    
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Record start time and memory usage
    start_time = time.time()
    timed_print(f"[Memory: {get_memory_usage():.1f}MB] Starting evaluation of model {model_name}...")
    
    # Adjust worker count to avoid resource shortage
    available_cpus = os.cpu_count() or 4
    memory_mb = get_memory_usage()
    
    # Dynamically adjust worker count
    if memory_mb > 4000:  # Memory usage exceeds 4GB
        adjusted_workers = min(workers, max(1, available_cpus // 2))
        if adjusted_workers < workers:
            timed_print(f"High memory usage, adjusting worker count: {workers} -> {adjusted_workers}")
            workers = adjusted_workers
    
    timed_print(f"Using {workers} worker threads for evaluation")
    
    # Load model data
    from data_loaders import load_model_data
    model_data = load_model_data(model_dir)
    if not model_data:
        timed_print(f"Error: Failed to load data for model {model_name}, skipping evaluation")
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
    
    # Record evaluated QA pairs
    total_qa_pairs = 0
    processed_qa_pairs = 0
    
    # Get all model scenarios (e.g. chat_general, routine)
    model_scenarios = list(model_data.keys())
    timed_print(f"Model data scenarios: {', '.join(model_scenarios)}")
    
    # Get all reference scenarios (e.g. Haru - SSR, Ryo-SSR)
    reference_scenarios = list(reference_data.keys())
    timed_print(f"Reference data scenarios: {', '.join(reference_scenarios)}")
    
    # Merge all reference answers into one dictionary, ignoring scenario distinction
    all_reference_responses = {}
    for scenario, scenario_data in reference_data.items():
        all_reference_responses.update(scenario_data)
    
    timed_print(f"Number of merged reference answers: {len(all_reference_responses)}")
    
    # Create worker thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        
        # Record start time
        eval_start = time.time()
        
        # Directly use each model scenario with merged reference data for matching
        for model_scenario in model_scenarios:
            scenario_start = time.time()
            
            # Get model responses
            model_responses = model_data[model_scenario]
            timed_print(f"Processing model scenario {model_scenario}: {len(model_responses)} responses")
            
            qa_pairs = []
            
            # Assign a reference answer to each model response
            for qa_id, model_response in model_responses.items():
                # Ensure model response is valid
                if not isinstance(model_response, str) or not model_response.strip():
                    continue
                
                # Assign reference answer to model response
                # All questions share the same reference answer set
                reference_qa_id = None
                ref_response = None
                
                # 1. If ID matches directly, use matching reference answer
                if qa_id in all_reference_responses:
                    reference_qa_id = qa_id
                    ref_response = all_reference_responses[qa_id]
                
                # 2. If using numeric ID, try to match numeric part
                elif qa_id.startswith('q'):
                    try:
                        num_id = int(qa_id[1:])
                        # Check different ID formats
                        potential_ids = [
                            str(num_id),  # Pure number
                            f"q{num_id}",  # q prefix
                            f"q{num_id:03d}"  # q prefix with zero padding
                        ]
                        
                        for potential_id in potential_ids:
                            if potential_id in all_reference_responses:
                                reference_qa_id = potential_id
                                ref_response = all_reference_responses[potential_id]
                                break
                    except ValueError:
                        pass
                
                # 3. If still no match, use first available reference answer
                if not ref_response and all_reference_responses:
                    reference_qa_id = next(iter(all_reference_responses.keys()))
                    ref_response = all_reference_responses[reference_qa_id]
                
                # For debugging
                if qa_id == "q001":
                    timed_print(f"Example - Model ID: {qa_id}, Reference ID: {reference_qa_id}, Model response length: {len(model_response)}, Reference answer length: {len(ref_response) if ref_response else 0}")
                
                # Add to evaluation queue
                if model_response and ref_response:
                    qa_pairs.append((qa_id, model_response, ref_response))
            
            total_qa_pairs += len(qa_pairs)
            timed_print(f"Scenario {model_scenario}: Found {len(qa_pairs)} evaluable QA pairs")
            
            # Submit evaluation tasks to thread pool
            for qa_id, model_response, ref_response in qa_pairs:
                future = executor.submit(process_qa_pair, qa_id, model_response, ref_response)
                futures.append((future, model_scenario, qa_id))
            
            scenario_time = time.time() - scenario_start
            if scenario_time > 1.0:
                timed_print(f"Setting up evaluation tasks for scenario {model_scenario} took: {scenario_time:.2f} seconds")
        
        # Collect results
        for future, scenario, qa_id in futures:
            try:
                metrics = future.result()
                if metrics:
                    processed_qa_pairs += 1
                    # Accumulate various metrics
                    for key in total_scores:
                        if key in metrics:
                            total_scores[key] += metrics[key]
            except Exception as e:
                timed_print(f"Error processing QA pair {qa_id}: {str(e)}")
    
    # Calculate average scores
    if processed_qa_pairs > 0:
        for key in total_scores:
            total_scores[key] /= processed_qa_pairs
    
    # Calculate composite score
    # Weights for different metrics
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
    
    # Calculate weighted average
    composite_score = 0.0
    for key, weight in weights.items():
        composite_score += total_scores[key] * weight
    
    # Add to results
    all_metrics = total_scores.copy()
    all_metrics["composite_score"] = composite_score
    
    # End timing
    total_eval_time = time.time() - start_time
    eval_only_time = time.time() - eval_start
    
    # Output evaluation results
    timed_print(f"Evaluation completed! Processed {processed_qa_pairs}/{total_qa_pairs} QA pairs")
    timed_print(f"Evaluation time: {eval_only_time:.2f} seconds, Total time: {total_eval_time:.2f} seconds")
    timed_print(f"Composite score: {composite_score:.4f}")
    
    for key, value in total_scores.items():
        timed_print(f"{key}: {value:.4f}")
    
    return all_metrics 