# Extractive Question Answering System with Ranked Answers & Confidence Scores

An end-to-end, modular **Extractive Question Answering (QA)** platform built with PyTorch, Hugging Face Transformers, and Flask. 

Given a **question** and **one or more context passages**, the system extracts relevant answer spans and returns a **ranked list of candidate answers**, each with a **calibrated confidence score**, most likely answer first. It supports multi-document evidence aggregation, unanswerable questions (SQuAD v2 abstention), dual inference engines (custom fine-tuned models and out-of-the-box Hugging Face pipelines), and an interactive web interface with span highlighting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## Key Features

- **Multi-Passage Candidate Ranking**: Extracts answer spans across multiple independent passages, pooling candidates into a single ranked list sorted by confidence.
- **Cross-Passage Calibrated Scoring**: Rather than unbounded raw logit sums, candidate spans are scored using normalized probabilities:
  $$\text{Confidence} = P(\text{start} = i) \times P(\text{end} = j) \in [0, 1]$$
  enabling direct score comparisons across different passages and documents.
- **Evidence Corroboration**: Answers repeated across multiple independent contexts are automatically merged with boosted aggregate confidence and a recorded `support` count.
- **SQuAD v2 Abstention**: Full support for unanswerable questions. When the provided context does not contain the answer, the system appropriately surfaces `[no answer in the provided context]`.
- **Sliding-Window Preprocessing**: Handles arbitrarily long context passages through overlapping sliding windows (`max_seq_length=384`, `stride=128`) without truncating passages or splitting answers.
- **Modular Flask Web Application**: Clean Application Factory architecture with Blueprints, strict validation schemas, centralized error formatting, and an isolated service layer.
- **Dual Inference Engines**: Seamlessly switch between a fine-tuned model checkpoint (`finetuned`) and an out-of-the-box Hugging Face baseline pipeline (`baseline`).
- **Interactive UI**: Modern, responsive single-page interface with dynamic passage management, real-time top-$k$ adjustment, engine toggling, and character-level passage highlighting.
- **RESTful JSON API**: Clean, versioned endpoints (`/api/health`, `/api/info`, `/api/answer`) for programmatic integration.

---

## Project Structure

```
extractive-qa-ranked/
├── app/                        # Modular Flask Application
│   ├── wsgi.py                 # WSGI entrypoint for development and Gunicorn
│   └── qa_web/                 # Flask application package
│       ├── __init__.py         # Application factory: create_app()
│       ├── settings.py         # Environment-driven configuration (Dev/Prod/Test)
│       ├── errors.py           # ApiError & centralized error handlers
│       ├── samples.py          # Built-in demo passages
│       ├── blueprints/
│       │   ├── web.py          # HTML route (Single-page UI)
│       │   └── api.py          # JSON API: /health, /info, /answer
│       ├── services/
│       │   ├── qa_service.py   # Model lifecycle & inference engine dispatch
│       │   └── validation.py   # Request schema parsing & guard rails
│       ├── templates/          # HTML templates (index.html)
│       └── static/             # Frontend assets (styles.css, app.js)
│
├── src/qa_system/              # Reusable ML Library
│   ├── config.py               # QAConfig: unified dataclass for all settings
│   ├── data.py                 # SQuAD data loading & Hugging Face datasets
│   ├── preprocessing.py        # Sliding-window tokenization & span alignment
│   ├── modeling.py             # Model/tokenizer loading, persistence, & devices
│   ├── baseline.py             # Out-of-the-box Hugging Face pipeline wrapper
│   ├── train.py                # Native PyTorch fine-tuning loop
│   ├── postprocessing.py       # Probability masking, ranking, & duplicate merge
│   ├── metrics.py              # EM/F1, Hit@k, Recall@k, MRR, & calibration
│   ├── evaluation.py           # Batched inference & evaluation reports
│   └── inference.py            # RankedQAPipeline serving engine
│
├── scripts/                    # Command-line utility scripts
│   ├── run_baseline.py         # Run inference with zero-training HF pipeline
│   ├── run_train.py            # Fine-tune encoder on SQuAD v2
│   ├── run_eval.py             # Evaluate checkpoint on span & ranking metrics
│   └── run_predict.py          # Run ranked predictions from the terminal
│
├── tests/                      # Unit & API contract test suite
│   ├── test_api.py             # Flask API contract tests (using stub service)
│   ├── test_validation.py      # Request validation tests
│   ├── test_postprocessing.py  # Ranking & confidence scoring tests
│   ├── test_preprocessing.py   # Sliding-window & offset mapping tests
│   └── test_metrics.py         # Evaluation metrics tests
│
├── data/                       # Sample SQuAD format datasets
├── run_webapp.py               # Top-level runner script
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

---

## Quickstart

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/teevvr7/Q-A_system.git
cd Q-A_system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application

Start the Flask development server:

```bash
python run_webapp.py
# or: python app/wsgi.py
```

The web UI is now accessible at **http://127.0.0.1:5000**.

> **Note**: If no fine-tuned model checkpoint is provided in `artifacts/qa-model` or specified via `QA_MODEL_DIR`, the application automatically falls back to `distilbert-base-cased-distilled-squad` so the service works out of the box with zero training required.

---

## API Reference

### Health Check

```http
GET /api/health
```

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_dir": "artifacts/qa-model"
}
```

### Model Info

```http
GET /api/info
```

**Response**:
```json
{
  "model": "artifacts/qa-model",
  "device": "cpu",
  "max_seq_length": 384,
  "doc_stride": 128,
  "default_top_k": 5,
  "max_top_k": 20,
  "aggregate_duplicates": true,
  "no_answer_supported": true,
  "no_answer_label": "[no answer in the provided context]"
}
```

### Answer Question

```http
POST /api/answer
Content-Type: application/json
```

**Request Body**:
```json
{
  "question": "How long is the limited warranty on the device?",
  "contexts": [
    "The device carries a two-year limited warranty against hardware defects.",
    "Accessories such as chargers and cables are covered for ninety days.",
    "Customers can optionally purchase an extended warranty for up to four years."
  ],
  "top_k": 3,
  "engine": "finetuned"
}
```

**Response**:
```json
{
  "question": "How long is the limited warranty on the device?",
  "answers": [
    {
      "rank": 1,
      "text": "ninety days",
      "confidence": 0.9578,
      "relative_score": 0.3925,
      "passage_index": 1,
      "start_char": 56,
      "end_char": 67,
      "support": 1,
      "passages": [1],
      "is_no_answer": false
    },
    {
      "rank": 2,
      "text": "two-year",
      "confidence": 0.9433,
      "relative_score": 0.3865,
      "passage_index": 0,
      "start_char": 21,
      "end_char": 29,
      "support": 1,
      "passages": [0],
      "is_no_answer": false
    },
    {
      "rank": 3,
      "text": "four years",
      "confidence": 0.5396,
      "relative_score": 0.2211,
      "passage_index": 2,
      "start_char": 65,
      "end_char": 75,
      "support": 1,
      "passages": [2],
      "is_no_answer": false
    }
  ],
  "top_answer": "ninety days",
  "n_passages": 3,
  "n_windows": 3,
  "top_k": 3,
  "latency_ms": 759.8,
  "engine": "finetuned",
  "model": "artifacts/qa-model",
  "error": null
}
```

---

## Configuration

Configuration is environment-driven via environment variables:

| Variable | Default | Description |
|---|---|---|
| `QA_ENV` | `dev` | Active environment: `dev`, `prod`, or `test` |
| `QA_MODEL_DIR` | `artifacts/qa-model` | Path to fine-tuned model checkpoint |
| `QA_FALLBACK_MODEL` | `distilbert-base-cased-distilled-squad` | Hugging Face model used if checkpoint is absent |
| `QA_BASELINE_MODEL` | `distilbert-base-cased-distilled-squad` | Model used for the baseline pipeline engine |
| `QA_EAGER_LOAD` | `0` | Set to `1` to warm up and load the model at startup |
| `QA_HOST` | `127.0.0.1` | Bind host address |
| `QA_PORT` | `5000` | Bind port |
| `QA_MAX_PASSAGES` | `20` | Maximum number of context passages per request |
| `QA_MAX_QUESTION_CHARS` | `1000` | Maximum character length for a question |
| `QA_MAX_PASSAGE_CHARS` | `50000` | Maximum character length per passage |

---

## CLI Utilities

The `scripts/` directory provides command-line interfaces for all pipeline operations:

- **Instant Prediction (Zero Training)**:
  ```bash
  python scripts/run_baseline.py \
    --question "When was Python released?" \
    --context "Python was created by Guido van Rossum and released in 1991."
  ```

- **Ranked Prediction Across Passages**:
  ```bash
  python scripts/run_predict.py --top-k 3 \
    --question "Where is the head office?" \
    --context "Head office is located in Bengaluru." \
    --context "Operations are handled across regional hubs."
  ```

- **Fine-Tuning**:
  ```bash
  python scripts/run_train.py \
    --model-name distilbert-base-uncased \
    --dataset-name squad_v2 \
    --num-train-epochs 2 \
    --train-batch-size 16 \
    --output-dir artifacts/qa-model
  ```

- **Evaluation**:
  ```bash
  python scripts/run_eval.py --model-dir artifacts/qa-model
  ```

---

## Running Tests

Run the comprehensive unit test suite covering validation, metrics, postprocessing, preprocessing, and API endpoints:

```bash
pytest tests/ -v
```

---

## Production Deployment

To serve the modular Flask application in production using Gunicorn:

```bash
pip install gunicorn
gunicorn --chdir app wsgi:app --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

---

## License

This project is licensed under the [MIT License](LICENSE).
