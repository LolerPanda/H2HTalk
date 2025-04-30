# H2HTalk: Evaluating Large Language Models as Emotional Companion

H2HTalk is a comprehensive benchmark for evaluating Large Language Models (LLMs) as emotional companions. It focuses on assessing models' capabilities in personality development and empathetic interaction, with a particular emphasis on emotional intelligence and conversational fluency.

## Project Overview

This project implements the evaluation framework described in the paper "H2HTalk: Evaluating Large Language Models as Emotional Companion". It provides tools and metrics for assessing LLMs' performance in providing emotional support and companionship.

## Key Features

- **Comprehensive Evaluation Framework**: Assesses models across multiple dimensions including personality development and empathetic interaction
- **Secure Attachment Persona (SAP) Framework**: Integrates attachment theory with interaction design
- **Multiple Evaluation Scenarios**: Includes dialogue, recollection, and itinerary scenarios
- **Rich Dataset**: Contains over 4,650 carefully crafted examples simulating emotional support dynamics
- **Diverse Metrics**: Implements various evaluation metrics including BLEU, ROUGE, BERTScore, and embedding-based similarity

## Project Structure

```
H2HTalk/
├── data/                    # Dataset files
│   ├── dialogue_*.jsonl     # Dialogue scenarios
│   ├── itinerary_*.jsonl    # Itinerary scenarios
│   └── recollection_*.jsonl # Recollection scenarios
├── evaluate/                # Evaluation framework
│   ├── core.py             # Core functionality
│   ├── evaluator.py        # Main evaluation logic
│   ├── metrics/            # Various evaluation metrics
│   └── visualizations.py   # Visualization tools
└── prompt_templet/         # Prompt templates
    ├── persona_setting/    # Persona configuration
    └── *.txt              # Various prompt templates
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your model outputs in the required format
2. Run the evaluation:
```bash
python evaluate/evaluate_model.py --model_outputs <path_to_outputs> --output_dir <results_dir>
```

## Evaluation Metrics

The framework includes multiple evaluation metrics:
- Text similarity metrics (BLEU, ROUGE)
- Semantic similarity (BERTScore)
- Embedding-based similarity
- Comprehensive scoring system