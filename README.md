# Autonomous APT Forensics & Impact Engine (Hybrid GraphRAG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%203050%20(6GB%20VRAM)%20%7C%2016GB%20RAM-green.svg)]()
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Graph Database](https://img.shields.io/badge/Graph-AWS%20FalkorDB-FF4B4B.svg)](https://www.falkordb.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An autonomous, senior-level cybersecurity intelligence system engineered to investigate multi-stage **Advanced Persistent Threats (APTs)**, reconstruct lateral movement chains, and compute exact compromise blast radiuses. 

By unifying **Semantic Vector Search (BGE-Small)** with **Structural Knowledge Graph Reasoning (FalkorDB openCypher)** and **LLM Graph Extraction (Qwen2.5-Coder-3B)**, the engine bridges the gap between static policy compliance and dynamic network causality—operating within strict edge hardware budgets (RTX 3050 6GB VRAM / 16GB RAM).

---

## 🏛️ System Architecture

```
                                      [ Raw Telemetry & Cloud Logs ]
                                                    │
                                                    ▼
                                      ┌───────────────────────────┐
                                      │   Qwen2.5-Coder-3B (4-bit)│  ◄── (Unsloth QLoRA Fine-Tuned)
                                      │   Causal Triple Extractor │
                                      └─────────────┬─────────────┘
                                                    │ (head, relation, tail)
                                                    ▼
                                      ┌───────────────────────────┐
                                      │    AWS FalkorDB Graph     │
                                      │ (Cypher Blast Radius Path)│
                                      └─────────────┬─────────────┘
                                                    │
[ CISO Query & Target IP ]                          │
           │                                        │
           ▼                                        │
┌─────────────────────┐                             │
│   FastAPI Router    │ ◄───────────────────────────┘
│ (/v1/threat/invest) │
└──────────┬──────────┘
           │
           ├────────────────────────► [ Semantic Vector DB (FAISS + BGE-Small) ]
           │                          (SEC-SOP Runbooks & Containment Policies)
           │
           ▼
┌────────────────────────────────────────────────────────┐
│            SentinelNode Streamlit Analyst UI           │
│  - Pyvis / Graphviz Interactive Blast Radius Visualizer│
│  - Gemini 2.5 Flash Executive Incident Response Report │
└────────────────────────────────────────────────────────┘
```

### Core Architecture Components:
1. **The Semantic Brain (Phase 1)**: Fine-tuned `BAAI/bge-small-en-v1.5` using contrastive learning (`MultipleNegativesRankingLoss`) to project security concepts, CVE IDs, and MITRE ATT&CK techniques into dense vector space under ~130MB VRAM footprint.
2. **The Structural Brain (Phase 2)**: Quantized 4-bit `Qwen/Qwen2.5-Coder-3B` fine-tuned via Unsloth QLoRA (`r=16`, `alpha=16`) to parse messy logs (SSH, NGINX, IAM) and extract structured causal triples `[head, relation, tail]`.
3. **The Hybrid Routing Engine (Phase 3)**: High-throughput FastAPI router that concurrently dispatches semantic queries to vector indexes while executing 1–3 hop variable-length Cypher traversals on AWS FalkorDB.
4. **SentinelNode Analyst UI (Phase 4)**: Streamlit interactive operations console featuring dynamic Pyvis graph visualization and automated Incident Response report generation via Google's `gemini-2.5-flash`.

---

## 📂 Repository Structure

| File Path | Description |
| :--- | :--- |
| `PROJECT_DIRECTORY.md` | Master index maintaining a single-sentence summary of every file in the project repository. |
| `.env` | Confidential environment variables (AWS FalkorDB endpoints, credentials, and Gemini API key). |
| `requirements.txt` | Production Python dependencies required to run the backend, client, and frontend. |
| `cyber_training_data.jsonl` | Contrastive training dataset mapping security entities to technical MITRE definitions. |
| `train_embeddings.py` | Embedding fine-tuning script utilizing Sentence-Transformers and MultipleNegativesRankingLoss. |
| `telemetry_graph_pairs.json` | Instruction-tuning dataset pairing raw server telemetry with structured causal graph triples. |
| `train_extractor.py` | Unsloth 4-bit QLoRA script fine-tuning Qwen2.5-Coder-3B within a 6GB VRAM budget. |
| `falkordb_client.py` | AWS FalkorDB client executing Cypher blast-radius graph traversals. |
| `main.py` | FastAPI application exposing `/v1/threat/investigate` for hybrid retrieval. |
| `streamlit_app.py` | Streamlit analyst interface rendering interactive blast-radius graphs and Gemini IR reports. |

---

## ⚡ Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ and an NVIDIA CUDA-enabled driver are installed.

```powershell
# Clone and enter workspace
git clone https://github.com/Atharva-Ramawat/autonomous-apt-graphrag.git
cd autonomous-apt-graphrag

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install standard dependencies
pip install -r requirements.txt
```

> **Note on Unsloth Installation**: Unsloth requires hardware-matched PyTorch wheels. For local fine-tuning on RTX 30-series GPUs, follow the official Unsloth instructions:
> ```powershell
> pip install --upgrade --no-cache-dir "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
> ```

---

### 2. Configure Environment Credentials (`.env`)
Create or edit your `.env` file in the repository root:

```env
# AWS FalkorDB Configuration
FALKORDB_HOST="your-falkordb-instance.amazonaws.com"
FALKORDB_PORT=6379
FALKORDB_PASSWORD="your-strong-password"
FALKORDB_GRAPH_NAME="apt_threat_graph"

# Google Gemini API
GEMINI_API_KEY="AIzaSyYourGeminiApiKey"
```

---

### 3. Running the Engine

#### Terminal 1: Launch FastAPI Hybrid Backend
```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*API Swagger Documentation will be live at `http://localhost:8000/docs`.*

#### Terminal 2: Launch SentinelNode Streamlit UI
```powershell
python -m streamlit run streamlit_app.py
```
*The web dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 🔬 Hardware Optimization Verification
- **Target GPU**: NVIDIA GeForce RTX 3050 (6GB VRAM)
- **Host Memory**: 16GB RAM
- **Embedding Footprint**: ~130MB VRAM via FP16 mixed precision.
- **Extractor Fine-Tuning**: Fits in ~4.2GB VRAM via 4-bit BitsAndBytes quantization, rank-16 QLoRA, and gradient checkpointing.
- **Inference Runtime**: Concurrently serves graph queries and vector matching in sub-second response times without GPU memory collisions.

---

## 📜 License
Distributed under the MIT License. Built for advanced cybersecurity operations and autonomous incident response.
