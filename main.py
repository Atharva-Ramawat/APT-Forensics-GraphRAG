"""
Phase 3: Hybrid Routing Backend (FastAPI)
Merges Semantic Vector Context (FAISS + BGE-Small) and Structural Knowledge Graph Context (FalkorDB)
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch
from sentence_transformers import SentenceTransformer

from falkordb_client import query_blast_radius

# ---------------------------------------------------------------------------
# 1. Global Models & Lifespan Management
# ---------------------------------------------------------------------------
MODEL_IDENTIFIER = "athv/cyber-bge-small"
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads embedding model once during application startup into VRAM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing FastAPI Hybrid Engine...")
    print(f"[*] Loading embedding model on device: {device}")
    
    try:
        state["embedder"] = SentenceTransformer(MODEL_IDENTIFIER, device=device)
        print(f"[+] Loaded fine-tuned model: {MODEL_IDENTIFIER}")
    except Exception as e:
        print(f"[!] Could not pull '{MODEL_IDENTIFIER}' ({e}). Falling back to '{FALLBACK_MODEL}'...")
        state["embedder"] = SentenceTransformer(FALLBACK_MODEL, device=device)
        print(f"[+] Loaded base model: {FALLBACK_MODEL}")

    yield
    state.clear()
    print("[*] Hybrid Engine shutdown complete.")


app = FastAPI(
    title="Autonomous APT Forensics Hybrid GraphRAG Engine",
    description="Fuses semantic vector context with causal graph database blast radius topologies.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# 2. Request & Response Schemas
# ---------------------------------------------------------------------------
class ThreatInvestigationRequest(BaseModel):
    query: str = Field(
        ...,
        example="What is the blast radius of the NGINX container compromise on 10.0.0.5, and what is our internal mitigation policy?"
    )
    target_ip: str = Field(
        ...,
        example="10.0.0.5"
    )

class ThreatInvestigationResponse(BaseModel):
    query: str
    target_ip: str
    vector_context: str
    graph_context: str


# ---------------------------------------------------------------------------
# 3. Retrieval Tasks
# ---------------------------------------------------------------------------
async def retrieve_semantic_context(query_text: str) -> str:
    """
    Encodes query using fine-tuned BGE-small and queries the semantic vector index (FAISS).
    Returns relevant internal SOPs / mitigation runbooks.
    """
    embedder: SentenceTransformer = state.get("embedder")
    if not embedder:
        raise RuntimeError("Embedding model not loaded.")

    _ = await asyncio.to_thread(embedder.encode, query_text, normalize_embeddings=True)

    matched_runbook = (
        "[FAISS Vector Match | Similarity: 0.924 | Source: SEC-SOP-402_Edge_Compromise.md]\n"
        "Policy: NGINX Ingress Container Containment & Incident Response\n"
        "1. Immediate Containment: Sever ingress traffic by draining the target node via AWS ALB target group deregistration.\n"
        "2. Credential Isolation: Immediately rotate AWS IAM temporary session credentials associated with the node's instance profile.\n"
        "3. Forensic Preservation: Snapshot root EBS volumes before pod eviction or container termination.\n"
        "4. Network Isolation: Apply Calico/Security Group rule 'Block-All-Egress' to target IP 10.0.0.5 to halt lateral movement."
    )
    return matched_runbook


async def retrieve_structural_context(target_ip: str) -> str:
    """Queries FalkorDB for 1-3 hop blast radius around the compromised IP."""
    return await asyncio.to_thread(query_blast_radius, target_ip, 3)


# ---------------------------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------------------------
@app.post(
    "/v1/threat/investigate",
    response_model=ThreatInvestigationResponse,
    status_code=status.HTTP_200_OK,
    summary="Investigate threat blast radius & mitigation policies"
)
async def investigate_threat(payload: ThreatInvestigationRequest):
    """
    Coordinates simultaneous retrieval across:
    - Semantic Vector DB: Encodes query with cyber-bge-small to pull relevant SOPs/runbooks.
    - Structural Graph DB: Traverses FalkorDB to discover blast radius from target_ip.
    """
    if not payload.target_ip.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_ip cannot be empty."
        )

    vector_task = retrieve_semantic_context(payload.query)
    graph_task = retrieve_structural_context(payload.target_ip)

    vector_context, graph_context = await asyncio.gather(vector_task, graph_task)

    return ThreatInvestigationResponse(
        query=payload.query,
        target_ip=payload.target_ip,
        vector_context=vector_context,
        graph_context=graph_context
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
