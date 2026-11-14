# Autonomous APT Forensics Hybrid GraphRAG Engine — Project Directory

| File Path | Description |
| :--- | :--- |
| `PROJECT_DIRECTORY.md` | Master index maintaining a single-sentence, plain-English summary of every file in the project repository. |
| `.gitignore` | Git exclusion manifest preventing confidential .env credentials, local model weights, checkpoints, and bytecode from entering version control. |
| `start_engine.ps1` | Master PowerShell automation script that installs Phase 2 dependencies, runs QLoRA training, and launches the FastAPI and Streamlit servers. |
| `README.md` | Comprehensive GitHub repository documentation detailing system architecture, hardware profiling, environment setup, and execution. |
| `requirements.txt` | Production dependency manifest defining required Python packages for vector embedding, graph querying, API routing, and UI rendering. |
| `.env` | Environment configuration file storing confidential credentials for the AWS FalkorDB instance and Google Gemini API. |
| `cyber_training_data.jsonl` | Labeled contrastive training dataset mapping cybersecurity threat entities, CVEs, and ATT&CK techniques to their technical definitions. |
| `train_embeddings.py` | Fine-tuning script utilizing Sentence-Transformers, MultipleNegativesRankingLoss, and mixed precision to adapt BAAI/bge-small-en-v1.5 under 6GB VRAM limits. |
| `telemetry_graph_pairs.json` | Instruction-tuning dataset containing raw, unstructured security telemetry paired with structured causal graph triples for extractor fine-tuning. |
| `train_extractor.py` | Unsloth-powered 4-bit QLoRA fine-tuning script adapting Qwen2.5-Coder-3B to extract causal knowledge graph triples within a 6GB VRAM budget. |
| `falkordb_client.py` | FalkorDB graph client that connects via environment variables and executes Cypher graph traversals to calculate blast radius around compromised network entities. |
| `main.py` | FastAPI application exposing `/v1/threat/investigate` to concurrently retrieve semantic runbook context via BGE-small and structural blast radius context via FalkorDB. |
| `streamlit_app.py` | Generative Streamlit analyst UI providing query inputs, FastAPI context retrieval, pyvis blast-radius graph visualization, and Gemini-2.5-Flash incident report generation. |
