#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module, provides report and chart generation functionality
"""

import os
import csv
import json
from core import timed_print
import time

def generate_visualizations(all_metrics, output_dir):
    """
    Generate visualization charts
    
    Args:
        all_metrics: Detailed evaluation metrics for all models, can be in list or dictionary format
        output_dir: Output directory
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create visualization directory
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Check input data format and convert to dictionary format
        if isinstance(all_metrics, list):
            # Convert list to dictionary
            all_metrics_dict = {}
            for model_data in all_metrics:
                if "model" in model_data:
                    all_metrics_dict[model_data["model"]] = model_data
                else:
                    timed_print("Warning: Model data missing 'model' field")
        else:
            all_metrics_dict = all_metrics
        
        # If no data, return
        if not all_metrics_dict:
            timed_print("Warning: No model data available for visualization")
            return
        
        # Extract evaluation metrics
        metrics_list = ["rouge1", "rouge2", "rougeL", "bertscore", 
                        "embedding_score", "bleu1", "bleu2", "bleu3", "bleu4"]
        
        # Sort by composite score and take top 10 models
        top_models = sorted(
            all_metrics_dict.items(), 
            key=lambda x: x[1].get("composite_score", 0), 
            reverse=True
        )[:10]
        
        if not top_models:
            timed_print("Warning: Not enough model data for visualization")
            return
        
        # 1. Radar chart: Compare different models' performance on various metrics
        try:
            plt.figure(figsize=(12, 10))
            
            # Set radar chart parameters
            angles = np.linspace(0, 2*np.pi, len(metrics_list), endpoint=False).tolist()
            angles += angles[:1]  # Close radar chart
            
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], metrics_list, size=12)
            
            # Draw radar chart for each model
            for i, (model_name, metrics) in enumerate(top_models):
                values = [metrics.get(metric, 0) for metric in metrics_list]
                values += values[:1]  # Close radar chart
                ax.plot(angles, values, linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            plt.title("Model Evaluation Metrics Radar Comparison", size=20)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Save radar chart
            radar_chart_path = os.path.join(visualizations_dir, "evaluation_results_radar_comparison.png")
            plt.savefig(radar_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            timed_print(f"Radar chart generated: {radar_chart_path}")
        except Exception as e:
            timed_print(f"Error generating radar chart: {e}")
        
        # 2. Bar chart: Compare composite scores of different models
        try:
            plt.figure(figsize=(12, 8))
            
            models = [model_name for model_name, _ in top_models]
            scores = [metrics.get("composite_score", 0) for _, metrics in top_models]
            
            # Reverse lists to show highest scoring model at top
            models.reverse()
            scores.reverse()
            
            plt.barh(models, scores, color='skyblue')
            plt.xlabel('Composite Score')
            plt.ylabel('Model Name')
            plt.title('Model Score Comparison')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Show specific scores
            for i, score in enumerate(scores):
                plt.text(score + 0.01, i, f'{score:.3f}', va='center')
            
            # Save bar chart
            score_chart_path = os.path.join(visualizations_dir, "evaluation_results_score_comparison.png")
            plt.savefig(score_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            timed_print(f"Score comparison chart generated: {score_chart_path}")
        except Exception as e:
            timed_print(f"Error generating score comparison chart: {e}")
        
        # 3. Heatmap: Detailed display of different models' performance on various metrics
        try:
            plt.figure(figsize=(14, 10))
            
            # Prepare heatmap data
            model_names = [model_name for model_name, _ in top_models]
            data = np.array([[metrics.get(metric, 0) for metric in metrics_list] 
                            for _, metrics in top_models])
            
            # Draw heatmap
            plt.imshow(data, cmap='YlGnBu', aspect='auto')
            plt.colorbar(label='Score')
            
            # Set labels
            plt.xticks(np.arange(len(metrics_list)), metrics_list, rotation=45)
            plt.yticks(np.arange(len(model_names)), model_names)
            
            # Show specific values in each cell
            for i in range(len(model_names)):
                for j in range(len(metrics_list)):
                    plt.text(j, i, f'{data[i, j]:.3f}', 
                            ha='center', va='center', 
                            color='black' if data[i, j] > 0.5 else 'white')
            
            plt.title('Model Evaluation Metrics Heatmap')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = os.path.join(visualizations_dir, "evaluation_results_metrics_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            timed_print(f"Metrics heatmap generated: {heatmap_path}")
        except Exception as e:
            timed_print(f"Error generating heatmap: {e}")
    
    except ImportError:
        timed_print("Warning: Generating visualizations requires matplotlib library, please install: pip install matplotlib")
    except Exception as e:
        timed_print(f"Error generating visualization charts: {e}")

def generate_report(all_metrics, output_dir):
    """
    Generate evaluation report
    
    Args:
        all_metrics: Composite scores for all models, can be in list or dictionary format
        output_dir: Output directory
    
    Returns:
        Path to generated report file
    """
    # Check input data format and convert to dictionary format
    if isinstance(all_metrics, list):
        # Convert list to dictionary
        all_metrics_dict = {}
        for model_data in all_metrics:
            if "model" in model_data:
                all_metrics_dict[model_data["model"]] = model_data
            else:
                timed_print("Warning: Model data missing 'model' field")
    else:
        all_metrics_dict = all_metrics
        
    # If no data, return
    if not all_metrics_dict:
        timed_print("Warning: No model data available for report generation")
        return None, None, None
        
    # Create model ranking table
    models_ranked = sorted(all_metrics_dict.items(), key=lambda x: x[1].get("composite_score", 0), reverse=True)
    
    # Prepare Markdown report
    md_report = "# Companion Model Evaluation Report/n\n"
    md_report += "## Model Rankings/n\n"
    
    # Create ranking table
    md_report += "| Rank | Model Name | Composite Score | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Embedding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |/n"
    md_report += "|------|------------|-----------------|---------|---------|---------|-----------|-----------|--------|--------|--------|--------|/n"
    
    # Create CSV report
    csv_rows = [["Rank", "Model Name", "Composite Score", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "Embedding", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]]
    
    for rank, (model_name, metrics) in enumerate(models_ranked, 1):
        # Get composite score
        composite_score = metrics.get("composite_score", 0.0)
        
        # Add to Markdown report
        md_report += f"| {rank} | {model_name} | {composite_score:.3f} | {metrics.get('rouge1', 0.0):.3f} | {metrics.get('rouge2', 0.0):.3f} | {metrics.get('rougeL', 0.0):.3f} | {metrics.get('bertscore', 0.0):.3f} | {metrics.get('embedding_score', 0.0):.3f} | {metrics.get('bleu1', 0.0):.3f} | {metrics.get('bleu2', 0.0):.3f} | {metrics.get('bleu3', 0.0):.3f} | {metrics.get('bleu4', 0.0):.3f} |/n"
        
        # Add to CSV report
        csv_rows.append([
            rank, 
            model_name, 
            f"{composite_score:.3f}", 
            f"{metrics.get('rouge1', 0.0):.3f}", 
            f"{metrics.get('rouge2', 0.0):.3f}", 
            f"{metrics.get('rougeL', 0.0):.3f}", 
            f"{metrics.get('bertscore', 0.0):.3f}", 
            f"{metrics.get('embedding_score', 0.0):.3f}", 
            f"{metrics.get('bleu1', 0.0):.3f}", 
            f"{metrics.get('bleu2', 0.0):.3f}", 
            f"{metrics.get('bleu3', 0.0):.3f}", 
            f"{metrics.get('bleu4', 0.0):.3f}"
        ])
    
    # Add evaluation metrics description
    md_report += "/n## Evaluation Metrics Description\n\n"
    md_report += "- **ROUGE-1/2/L**: Metric to measure the similarity between model generated text and data text in word (1-gram/2-gram) or longest common subsequence matching/n"
    md_report += "- **BERTScore**: Semantic similarity score based on BERT, better for capturing semantic similarity/n"
    md_report += "- **Embedding**: Semantic similarity score calculated using text embedding model, for capturing deep semantic understanding/n"
    md_report += "- **BLEU-1/2/3/4**: Metric to measure the similarity between generated text and data text in n-gram overlap/n"
    md_report += "- **Composite Score**: Weighted average score considering all above metrics/n\n"
    
    # Add scoring weight description
    md_report += "## Scoring Weights/n\n"
    md_report += "Composite score calculation uses the following weight allocation:/n"
    md_report += "- ROUGE-1: 20%/n"
    md_report += "- ROUGE-2: 10%/n"
    md_report += "- ROUGE-L: 20%/n"
    md_report += "- BERTScore: 25%/n"
    md_report += "- Embedding: 15%/n"
    md_report += "- BLEU-1: 5%/n"
    md_report += "- BLEU-4: 5%/n"
    
    # Save Markdown report
    md_report_path = os.path.join(output_dir, "report.md")
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    # Save CSV report
    csv_report_path = os.path.join(output_dir, "report.csv")
    with open(csv_report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    # Save JSON format report (for later processing)
    json_report_path = os.path.join(output_dir, "report.json")
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics_dict, f, ensure_ascii=False, indent=2)
    
    timed_print(f"Evaluation report generated: {md_report_path}")
    timed_print(f"CSV report generated: {csv_report_path}")
    timed_print(f"JSON report generated: {json_report_path}")
    
    return md_report_path, csv_report_path, json_report_path

def generate_html_report(all_metrics, output_dir):
    """
    Generate HTML format evaluation report
    
    Args:
        all_metrics: Detailed evaluation metrics for all models, can be in list or dictionary format
        output_dir: Output directory
        
    Returns:
        HTML report file path
    """
    try:
        # Check input data format and convert to dictionary format
        if isinstance(all_metrics, list):
            # Convert list to dictionary
            all_metrics_dict = {}
            for model_data in all_metrics:
                if "model" in model_data:
                    all_metrics_dict[model_data["model"]] = model_data
                else:
                    timed_print("Warning: Model data missing 'model' field")
        else:
            all_metrics_dict = all_metrics
        
        # If no data, return
        if not all_metrics_dict:
            timed_print("Warning: No model data available for generating HTML report")
            return None
        
        # Create HTML report directory
        html_dir = os.path.join(output_dir, "html_report")
        os.makedirs(html_dir, exist_ok=True)
        
        # Visualizations directory
        visualizations_dir = os.path.join(output_dir, "visualizations")
        
        # Prepare HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Companion Model Evaluation Report</title>
            <style>
                body { font-family: 'Arial', sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
                h1, h2 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #f5f5f5; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                tr:hover { background-color: #f1f1f1; }
                .metric-description { background-color: #f9f9ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .charts { display: flex; flex-direction: column; align-items: center; gap: 20px; margin-top: 20px; }
                img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
                .model-rank-1 { background-color: #fdf2e9 !important; }
                .model-rank-2 { background-color: #f9f9f9 !important; }
                .model-rank-3 { background-color: #f0f9ff !important; }
                footer { margin-top: 30px; text-align: center; font-size: 0.9em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <h1>Companion Model Evaluation Report</h1>
            
            <h2>Model Rankings</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model Name</th>
                    <th>Composite Score</th>
                    <th>ROUGE-1</th>
                    <th>ROUGE-2</th>
                    <th>ROUGE-L</th>
                    <th>BERTScore</th>
                    <th>Embedding</th>
                    <th>BLEU-1</th>
                    <th>BLEU-2</th>
                    <th>BLEU-3</th>
                    <th>BLEU-4</th>
                </tr>
        """
        
        # Add data row for each model
        models_ranked = sorted(all_metrics_dict.items(), key=lambda x: x[1].get("composite_score", 0), reverse=True)
        for rank, (model_name, metrics) in enumerate(models_ranked, 1):
            composite_score = metrics.get("composite_score", 0.0)
            
            # Add special styles for top 3 models
            row_class = f'class="model-rank-{rank}"' if rank <= 3 else ""
            
            html_content += f"""
                <tr {row_class}>
                    <td>{rank}</td>
                    <td>{model_name}</td>
                    <td><strong>{composite_score:.3f}</strong></td>
                    <td>{metrics.get('rouge1', 0.0):.3f}</td>
                    <td>{metrics.get('rouge2', 0.0):.3f}</td>
                    <td>{metrics.get('rougeL', 0.0):.3f}</td>
                    <td>{metrics.get('bertscore', 0.0):.3f}</td>
                    <td>{metrics.get('embedding_score', 0.0):.3f}</td>
                    <td>{metrics.get('bleu1', 0.0):.3f}</td>
                    <td>{metrics.get('bleu2', 0.0):.3f}</td>
                    <td>{metrics.get('bleu3', 0.0):.3f}</td>
                    <td>{metrics.get('bleu4', 0.0):.3f}</td>
                </tr>
            """
        
        # Add evaluation metrics description
        html_content += """
            </table>
            
            <h2>Evaluation Metrics Description</h2>
            <div class="metric-description">
                <p><strong>ROUGE-1/2/L</strong>: Metric to measure the similarity between model generated text and data text in word (1-gram/2-gram) or longest common subsequence matching</p>
                <p><strong>BERTScore</strong>: Semantic similarity score based on BERT, better for capturing semantic similarity</p>
                <p><strong>Embedding</strong>: Semantic similarity score calculated using text embedding model, for capturing deep semantic understanding</p>
                <p><strong>BLEU-1/2/3/4</strong>: Metric to measure the similarity between generated text and data text in n-gram overlap</p>
                <p><strong>Composite Score</strong>: Weighted average score considering all above metrics</p>
            </div>
            
            <h2>Scoring Weights</h2>
            <div class="metric-description">
                <p>Composite score calculation uses the following weight allocation:</p>
                <ul>
                    <li>ROUGE-1: 20%</li>
                    <li>ROUGE-2: 10%</li>
                    <li>ROUGE-L: 20%</li>
                    <li>BERTScore: 25%</li>
                    <li>Embedding: 15%</li>
                    <li>BLEU-1: 5%</li>
                    <li>BLEU-4: 5%</li>
                </ul>
            </div>
        """
        
        # Try to add visualization charts
        if os.path.exists(visualizations_dir):
            html_content += """
            <h2>Visualization Results</h2>
            <div class="charts">
            """
            
            # Check each chart file exists and add
            charts = [
                ("evaluation_results_radar_comparison.png", "Model Evaluation Metrics Radar Comparison"),
                ("evaluation_results_score_comparison.png", "Model Composite Score Comparison"),
                ("evaluation_results_metrics_heatmap.png", "Model Evaluation Metrics Heatmap")
            ]
            
            for chart_file, chart_title in charts:
                chart_path = os.path.join(visualizations_dir, chart_file)
                if os.path.exists(chart_path):
                    html_content += f"""
                <div>
                    <h3>{chart_title}</h3>
                    <img src="{chart_file}" alt="{chart_title}">
                </div>
                    """
            
            html_content += """
            </div>
            """
        
        # Add footer
        html_content += """
            <footer>
                <p>Generated time: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p>Companion Evaluation Metrics Tool</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        html_report_path = os.path.join(html_dir, "index.html")
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # If visualizations directory exists, copy images to HTML directory
        if os.path.exists(visualizations_dir):
            import shutil
            for file in os.listdir(visualizations_dir):
                if file.endswith('.png'):
                    shutil.copy(
                        os.path.join(visualizations_dir, file),
                        os.path.join(html_dir, file)
                    )
        
        print(f"HTML report generated: {html_report_path}")
        return html_report_path
        
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        return None 