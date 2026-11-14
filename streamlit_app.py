"""
Phase 4: SentinelNode Analyst UI (Streamlit)
Interactive Generative Dashboard for CISO Threat Investigation
Fuses FastAPI Hybrid Context with Gemini-2.5-Flash & Graph Visualization
"""

import os
import re
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="SentinelNode | Autonomous APT Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ SentinelNode Config")
st.sidebar.markdown("Configure backend communication & AI API credentials.")

backend_url = st.sidebar.text_input(
    "FastAPI Backend Endpoint",
    value=os.getenv("BACKEND_URL", "http://localhost:8000/v1/threat/investigate")
)

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Architecture:**\n"
    "- **Semantic DB:** BGE-Small (Contrastive Fine-Tuned)\n"
    "- **Structural DB:** AWS FalkorDB (Cypher Traversal)\n"
    "- **Executive Synthesis:** Google Gemini 2.5 Flash\n"
    "- **Local GPU Limit:** RTX 3050 (6GB VRAM)"
)

# ---------------------------------------------------------------------------
# Helper: Visual Graph Generator (Pyvis with Graphviz fallback)
# ---------------------------------------------------------------------------
def render_blast_radius_graph(graph_context: str, target_ip: str):
    """
    Parses topology from the graph_context string and renders an interactive
    network graph using Pyvis (or falls back to built-in Graphviz).
    """
    # Extract nodes and relationships using regex or pattern matching
    edges = []
    nodes = {target_ip: {"label": f"TARGET IP\n{target_ip}", "color": "#EF4444", "shape": "box"}}

    # Match patterns like: (Entity) -[relation]-> (Target) or └── -[rel]-> (Target)
    pattern = re.compile(r"\(?(?:Entity:\s*)?([^\)]+?)\)?\s*->?\s*-\[([a-zA-Z0-9_]+)\]->?\s*\(?(?:Host|Service|Role|Entity:\s*)?([^\)\n]+)\)?")
    matches = pattern.findall(graph_context)

    for src, rel, dst in matches:
        src = src.strip().replace("└──", "").replace("-", "").strip()
        dst = dst.strip().replace("└──", "").replace("-", "").strip()
        if not src:
            src = target_ip
        if src not in nodes:
            nodes[src] = {"label": src, "color": "#3B82F6", "shape": "ellipse"}
        if dst not in nodes:
            dst_color = "#F59E0B" if "Role" in dst or "Access" in dst or "admin" in dst.lower() else "#10B981"
            nodes[dst] = {"label": dst, "color": dst_color, "shape": "ellipse"}
        edges.append((src, dst, rel))

    # If simple pattern didn't capture line-based mock format, populate standard mock topology
    if not edges:
        nodes = {
            f"IP_{target_ip}": {"label": f"COMPROMISED IP\n{target_ip}", "color": "#EF4444", "shape": "box"},
            "NGINX_Gateway": {"label": "Host: NGINX_Gateway\n10.0.0.5", "color": "#F97316", "shape": "ellipse"},
            "Auth_Service": {"label": "Service: Auth_Service\n10.0.1.14", "color": "#3B82F6", "shape": "ellipse"},
            "RDS_ReadOnly_Access": {"label": "IAM Role:\nRDS_ReadOnly_Access", "color": "#8B5CF6", "shape": "diamond"},
            "Prod_Postgres_DB": {"label": "Database:\nProd_Customer_DB", "color": "#EC4899", "shape": "database"}
        }
        edges = [
            (f"IP_{target_ip}", "NGINX_Gateway", "failed_auth / ingress"),
            ("NGINX_Gateway", "Auth_Service", "lateral_connection"),
            ("Auth_Service", "RDS_ReadOnly_Access", "assumed_role"),
            ("RDS_ReadOnly_Access", "Prod_Postgres_DB", "db_read_access")
        ]

    # Try Pyvis interactive render
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components

        net = Network(height="420px", width="100%", bgcolor="#0F172A", font_color="#F8FAFC", directed=True)
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

        for node_id, attrs in nodes.items():
            net.add_node(node_id, label=attrs["label"], color=attrs["color"], shape=attrs.get("shape", "ellipse"))

        for src, dst, label in edges:
            net.add_edge(src, dst, title=label, label=label, color="#94A3B8")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_path = tmp_file.name

        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=440)
        os.remove(tmp_path)

    except ImportError:
        # Fallback to Graphviz (native in Streamlit)
        import graphviz
        dot = graphviz.Digraph(comment="Blast Radius")
        dot.attr(bgcolor="#0F172A")
        dot.attr("node", style="filled", fontcolor="white", shape="ellipse")
        dot.attr("edge", fontcolor="#94A3B8", color="#64748B")

        for node_id, attrs in nodes.items():
            dot.node(node_id, attrs["label"], fillcolor=attrs["color"])
        for src, dst, label in edges:
            dot.edge(src, dst, label=label)

        st.graphviz_chart(dot)


# ---------------------------------------------------------------------------
# Helper: Gemini 2.5 Flash Executive Synthesis
# ---------------------------------------------------------------------------
def generate_incident_report(api_key: str, query: str, target_ip: str, vector_context: str, graph_context: str) -> str:
    """Uses Gemini 2.5 Flash to synthesize fused GraphRAG context into a formal IR report."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    system_prompt = f"""You are SentinelNode, a Principal Incident Responder and Threat Forensics Specialist.
A CISO has submitted an urgent incident investigation query:
"{query}"

Target Compromised IP: {target_ip}

You are provided with two verified ground-truth context feeds from our Hybrid GraphRAG pipeline:

=== 1. SEMANTIC CONTEXT (Vector DB Runbooks & Enterprise Security Policies) ===
{vector_context}

=== 2. STRUCTURAL CONTEXT (FalkorDB Blast Radius & Topological Traversal) ===
{graph_context}

Produce a formal, highly technical Incident Response Briefing in structured Markdown:
1. **Executive Summary & Threat Attribution**: Impact level, primary compromise vector, and MITRE ATT&CK technique mapping.
2. **Blast Radius & Topology Assessment**: Clear breakdown of compromised assets, lateral movement routes, and high-value target (HVT) exposure.
3. **Immediate Containment Runbook**: Actionable step-by-step commands/procedures strictly reflecting the retrieved runbook.
4. **Forensic Preservation & Root Cause Analysis**: Steps for evidence collection (snapshots, logs) and egress hardening.
"""

    response = model.generate_content(system_prompt)
    return response.text


# ---------------------------------------------------------------------------
# Main UI Presentation
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ SentinelNode Analyst Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous APT Forensics & Blast Radius Investigation Engine (Hybrid GraphRAG)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_area(
        "CISO Investigation Query",
        value="What is the blast radius of the NGINX container compromise on 10.0.0.5, and what is our internal mitigation policy?",
        height=100
    )
with col2:
    target_ip = st.text_input(
        "Compromised Target IP",
        value="10.0.0.5"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    investigate_btn = st.button("🚀 Investigate Threat", use_container_width=True, type="primary")

if investigate_btn:
    if not gemini_api_key.strip():
        st.error("⚠️ Please provide a Gemini API Key in the sidebar to generate the executive report.")
        st.stop()

    with st.spinner("Querying Hybrid GraphRAG Engine (FastAPI -> BGE-Small + FalkorDB)..."):
        try:
            payload = {"query": user_query, "target_ip": target_ip}
            res = requests.post(backend_url, json=payload, timeout=20)

            if res.status_code != 200:
                st.error(f"❌ Backend error ({res.status_code}): {res.text}")
                st.stop()

            data = res.json()
            vector_context = data.get("vector_context", "")
            graph_context = data.get("graph_context", "")

        except requests.exceptions.ConnectionError:
            st.warning("⚠️ Could not connect to FastAPI on localhost:8000. Running in local demonstration fallback mode.")
            vector_context = (
                "[FAISS Vector Match | SEC-SOP-402: Container Isolation & Egress Severing]\n"
                "1. Egress Severing: Drop all outbound TCP traffic on 10.0.0.5 via AWS Security Group.\n"
                "2. IAM Isolation: Revoke active session tokens for associated instance profiles.\n"
                "3. Root Snapshot: Trigger immediate AWS EBS volume snapshot before container eviction."
            )
            graph_context = (
                f"Blast Radius for IP: {target_ip} (3 hops):\n"
                f"└── (IP_{target_ip}) -[failed_auth]-> (Host_NGINX_10.0.0.5)\n"
                f"    └── -[lateral_movement]-> (Service_Auth_10.0.1.14)\n"
                f"        └── -[assumed_role]-> (Role_RDS_ReadOnly_Access)"
            )

    # Display Graph & Raw Retreived Telemetry
    st.markdown("### 🌐 Dynamic Blast Radius Graph")
    render_blast_radius_graph(graph_context, target_ip)

    with st.expander("🔍 View Raw Hybrid Context (Vector + Graph)"):
        v_col, g_col = st.columns(2)
        with v_col:
            st.markdown("**Semantic Context (FAISS / BGE-Small):**")
            st.code(vector_context, language="markdown")
        with g_col:
            st.markdown("**Structural Context (FalkorDB / Cypher):**")
            st.code(graph_context, language="markdown")

    # Generate Executive Report with Gemini
    st.markdown("---")
    st.markdown("### 📋 Executive Incident Response Report")
    with st.spinner("Synthesizing Executive Incident Response Report via Gemini 2.5 Flash..."):
        try:
            report_markdown = generate_incident_report(
                api_key=gemini_api_key,
                query=user_query,
                target_ip=target_ip,
                vector_context=vector_context,
                graph_context=graph_context
            )
            st.markdown(report_markdown)
            st.download_button(
                label="📥 Download Incident Response Brief (.md)",
                data=report_markdown,
                file_name=f"Incident_Report_{target_ip}.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"❌ Gemini Generation failed: {str(e)}")
