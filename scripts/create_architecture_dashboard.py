"""
Create an HTML dashboard comparing neural network architecture performance.

Parses experiments/architecture_runs.txt and generates visualizations.
"""

import re
from pathlib import Path
from datetime import datetime
import json

try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent
except Exception:
    proj_root = Path("/home/ksmith/birds/neural_networks/fraud_detection")


def parse_architecture_runs(filepath):
    """Parse the architecture runs file and extract metrics for each run."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split by the separator lines
    runs = content.split("=" * 80)

    results = []
    for run in runs:
        if not run.strip():
            continue

        # Extract model name
        model_match = re.search(r"Model: (\w+)", run)
        if not model_match:
            continue
        model_name = model_match.group(1)

        # Extract training config
        epochs_match = re.search(r"epochs: (\d+)", run)
        lr_match = re.search(r"learning_rate: ([\d.]+)", run)

        # Extract metrics
        test_loss = re.search(r"test_loss: ([\d.]+)", run)
        test_precision = re.search(r"test_precision: ([\d.]+)", run)
        test_recall = re.search(r"test_recall: ([\d.]+)", run)
        test_f1 = re.search(r"test_f1: ([\d.]+)", run)
        test_roc_auc = re.search(r"test_roc_auc: ([\d.]+)", run)

        # Extract datetime
        datetime_match = re.search(r"Run datetime: ([\d\-: .]+)", run)

        if all([test_loss, test_precision, test_recall, test_f1, test_roc_auc]):
            result = {
                "model": model_name,
                "epochs": int(epochs_match.group(1)) if epochs_match else None,
                "learning_rate": float(lr_match.group(1))
                if lr_match
                else 0.008,
                "test_loss": float(test_loss.group(1)),
                "test_precision": float(test_precision.group(1)),
                "test_recall": float(test_recall.group(1)),
                "test_f1": float(test_f1.group(1)),
                "test_roc_auc": float(test_roc_auc.group(1)),
                "datetime": datetime_match.group(1) if datetime_match else None,
            }
            results.append(result)

    return results


def generate_html_dashboard(results, output_path):
    """Generate an interactive HTML dashboard with Plotly charts."""

    # Group results by model and select best run for each
    model_best = {}
    model_all = {}

    for result in results:
        model = result["model"]
        if model not in model_all:
            model_all[model] = []
        model_all[model].append(result)

        # Keep the best F1 score for each model
        if (
            model not in model_best
            or result["test_f1"] > model_best[model]["test_f1"]
        ):
            model_best[model] = result

    # Architecture parameter counts (approximate)
    param_counts = {
        "BaselineFraudNN": 4_700,
        "WideFraudNN": 30_900,
        "DeepFraudNN": 17_600,
        "ResNetFraudNN": 17_600,
        "BatchNormFraudNN": 17_600,
    }

    # Model display names
    display_names = {
        "BaselineFraudNN": "Baseline (5 layers)",
        "WideFraudNN": "Wide (5 layers, 128 units)",
        "DeepFraudNN": "Deep (11 layers)",
        "ResNetFraudNN": "ResNet (11 layers + skip)",
        "BatchNormFraudNN": "BatchNorm (11 layers + BN)",
    }

    # Generate Plotly charts data
    models = list(model_best.keys())
    model_labels = [display_names.get(m, m) for m in models]

    f1_scores = [model_best[m]["test_f1"] for m in models]
    precision_scores = [model_best[m]["test_precision"] for m in models]
    recall_scores = [model_best[m]["test_recall"] for m in models]
    roc_auc_scores = [model_best[m]["test_roc_auc"] for m in models]
    loss_scores = [model_best[m]["test_loss"] for m in models]
    params = [param_counts.get(m, 0) for m in models]

    # Color coding: green for good, yellow for ok, red for failed
    colors = []
    for m in models:
        f1 = model_best[m]["test_f1"]
        if f1 >= 0.75:
            colors.append("#2ecc71")  # Green - good
        elif f1 >= 0.40:
            colors.append("#f39c12")  # Orange - ok
        else:
            colors.append("#e74c3c")  # Red - failed

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Architecture Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .chart {{
            margin-bottom: 40px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        .summary-table th {{
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .summary-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .summary-table tr:hover {{
            background: #f8f9fa;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-excellent {{ background: #2ecc71; color: white; }}
        .status-ok {{ background: #f39c12; color: white; }}
        .status-failed {{ background: #e74c3c; color: white; }}
        .insight-box {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .insight-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Neural Network Architecture Comparison</h1>
        <p class="subtitle">Fraud Detection Performance Analysis - Generated {
        datetime.now().strftime("%Y-%m-%d %H:%M")
    }</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Architectures Tested</h3>
                <div class="value">{len(models)}</div>
            </div>
            <div class="metric-card">
                <h3>Total Experiments</h3>
                <div class="value">{len(results)}</div>
            </div>
            <div class="metric-card">
                <h3>Best F1 Score</h3>
                <div class="value">{max(f1_scores):.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Best ROC-AUC</h3>
                <div class="value">{max(roc_auc_scores):.3f}</div>
            </div>
        </div>

        <div class="chart" id="f1-comparison"></div>
        <div class="chart" id="metrics-comparison"></div>
        <div class="chart" id="params-vs-performance"></div>
        <div class="chart" id="precision-recall"></div>

        <h2>Detailed Results Summary</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Architecture</th>
                    <th>Params</th>
                    <th>Epochs</th>
                    <th>LR</th>
                    <th>F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>ROC-AUC</th>
                    <th>Loss</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {
        "".join(
            [
                f'''
                <tr>
                    <td><strong>{display_names.get(m, m)}</strong></td>
                    <td>{param_counts.get(m, 0):,}</td>
                    <td>{model_best[m]['epochs']}</td>
                    <td>{model_best[m]['learning_rate']}</td>
                    <td>{model_best[m]['test_f1']:.3f}</td>
                    <td>{model_best[m]['test_precision']:.3f}</td>
                    <td>{model_best[m]['test_recall']:.3f}</td>
                    <td>{model_best[m]['test_roc_auc']:.3f}</td>
                    <td>{model_best[m]['test_loss']:.3f}</td>
                    <td><span class="status-badge status-{'excellent' if model_best[m]['test_f1'] >= 0.75 else 'ok' if model_best[m]['test_f1'] >= 0.40 else 'failed'}">
                        {'✅ Excellent' if model_best[m]['test_f1'] >= 0.75 else '⚠️ OK' if model_best[m]['test_f1'] >= 0.40 else '❌ Failed'}
                    </span></td>
                </tr>
                '''
                for m in models
            ]
        )
    }
            </tbody>
        </table>

        <div class="insight-box">
            <h3>💡 Key Findings</h3>
            <ul>
                <li><strong>Baseline</strong> ({
        param_counts.get(
            "BaselineFraudNN", 0
        ):,} params): Simple 5-layer network, stable training, F1={
        model_best.get("BaselineFraudNN", {}).get("test_f1", 0):.3f}</li>
                <li><strong>Deep Network</strong> ({
        param_counts.get(
            "DeepFraudNN", 0
        ):,} params): Failed due to vanishing gradients (F1={
        model_best.get("DeepFraudNN", {}).get(
            "test_f1", 0
        ):.3f}, ROC-AUC≈0.5 = random)</li>
                <li><strong>ResNet</strong> ({
        param_counts.get(
            "ResNetFraudNN", 0
        ):,} params): Skip connections solved vanishing gradients but training unstable (explodes after epoch 7)</li>
                <li><strong>Wider Network</strong> ({
        param_counts.get(
            "WideFraudNN", 0
        ):,} params): 6.5× more parameters, sensitive to learning rate</li>
                <li><strong>Best Performer</strong>: {
        max(model_best.items(), key=lambda x: x[1]["test_f1"])[0]
    } with F1={max(f1_scores):.3f}</li>
            </ul>
        </div>
    </div>

    <script>
        // F1 Score Comparison
        var f1Data = [{{
            x: {json.dumps(model_labels)},
            y: {json.dumps(f1_scores)},
            type: 'bar',
            marker: {{
                color: {json.dumps(colors)},
            }},
            text: {json.dumps([f"{f:.3f}" for f in f1_scores])},
            textposition: 'auto',
        }}];

        var f1Layout = {{
            title: 'F1 Score Comparison (Higher is Better)',
            yaxis: {{ title: 'F1 Score', range: [0, 1] }},
            xaxis: {{ title: 'Architecture' }},
            height: 400,
        }};

        Plotly.newPlot('f1-comparison', f1Data, f1Layout);

        // Multi-metric Comparison
        var metricsData = [
            {{
                x: {json.dumps(model_labels)},
                y: {json.dumps(precision_scores)},
                name: 'Precision',
                type: 'bar',
            }},
            {{
                x: {json.dumps(model_labels)},
                y: {json.dumps(recall_scores)},
                name: 'Recall',
                type: 'bar',
            }},
            {{
                x: {json.dumps(model_labels)},
                y: {json.dumps(roc_auc_scores)},
                name: 'ROC-AUC',
                type: 'bar',
            }}
        ];

        var metricsLayout = {{
            title: 'Precision, Recall, and ROC-AUC Comparison',
            yaxis: {{ title: 'Score', range: [0, 1] }},
            xaxis: {{ title: 'Architecture' }},
            barmode: 'group',
            height: 400,
        }};

        Plotly.newPlot('metrics-comparison', metricsData, metricsLayout);

        // Parameters vs Performance
        var paramsData = [{{
            x: {json.dumps(params)},
            y: {json.dumps(f1_scores)},
            mode: 'markers+text',
            type: 'scatter',
            text: {json.dumps(model_labels)},
            textposition: 'top center',
            marker: {{
                size: 15,
                color: {json.dumps(colors)},
            }},
        }}];

        var paramsLayout = {{
            title: 'Model Complexity vs Performance',
            xaxis: {{ title: 'Parameters', type: 'log' }},
            yaxis: {{ title: 'F1 Score', range: [0, 1] }},
            height: 400,
        }};

        Plotly.newPlot('params-vs-performance', paramsData, paramsLayout);

        // Precision-Recall Tradeoff
        var prData = [{{
            x: {json.dumps(recall_scores)},
            y: {json.dumps(precision_scores)},
            mode: 'markers+text',
            type: 'scatter',
            text: {json.dumps(model_labels)},
            textposition: 'top center',
            marker: {{
                size: 15,
                color: {json.dumps(colors)},
            }},
        }}];

        var prLayout = {{
            title: 'Precision-Recall Tradeoff',
            xaxis: {{ title: 'Recall (Fraud Caught)', range: [0, 1] }},
            yaxis: {{ title: 'Precision (Accuracy When Predicting Fraud)', range: [0, 1] }},
            height: 400,
            shapes: [{{
                type: 'line',
                x0: 0, y0: 0, x1: 1, y1: 1,
                line: {{ color: 'gray', dash: 'dash' }}
            }}]
        }};

        Plotly.newPlot('precision-recall', prData, prLayout);
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_template)

    print(f"✅ Dashboard created: {output_path}")
    print(
        f"📊 Analyzed {len(results)} experiments across {len(models)} architectures"
    )


if __name__ == "__main__":
    runs_file = proj_root / "experiments/architecture_runs.txt"
    output_file = proj_root / "architecture_dashboard.html"

    if not runs_file.exists():
        print(f"❌ Error: {runs_file} not found")
        exit(1)

    results = parse_architecture_runs(runs_file)

    if not results:
        print("❌ No results found in the runs file")
        exit(1)

    generate_html_dashboard(results, output_file)
    print(f"\n🌐 Open in browser: file://{output_file}")
