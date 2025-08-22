import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🎓", layout="wide")

# Sidebar: Recent and Settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"- Uses backend: `{BACKEND_URL}`")
    st.divider()
    st.header("🗂️ Recent Reports")
    try:
        resp = requests.get(f"{BACKEND_URL}/list-reports", timeout=10)
        resp.raise_for_status()
        recent = resp.json()
        for r in (recent or [])[:7]:
            title = r.get("title") or "(untitled)"
            rid = r.get("id", "")[:8]
            st.write(f"- {title} ({rid}…)")
    except Exception as e:
        st.caption(f"(could not load: {e})")

st.title("🎓 Multi-Agent Research Assistant")
tabs = st.tabs(["Research", "Search"])

def toast_success(msg: str):
    st.toast(msg, icon="✅")

def toast_error(msg: str):
    st.toast(msg, icon="❌")

# ========== Research Tab (Streaming Only) ==========
with tabs[0]:
    st.subheader("Generate a new report (Streaming)")
    topic = st.text_input("Topic", placeholder="e.g., Vector databases for RAG")
    start_btn = st.button("Start Streaming", use_container_width=True)

    stage_box = st.container(border=True)
    output_box = st.container(border=True)

    if start_btn and topic.strip():
        stage_placeholder = stage_box.empty()
        output_placeholder = output_box.empty()

        stage_status = {"researcher": "pending", "analyst": "pending", "writer": "pending"}

        def render_stages():
            def badge(name, status):
                color = {"pending": "gray", "running": "orange", "done": "green"}.get(status, "gray")
                return f"<span style='color:{color};font-weight:600'>{name}: {status}</span>"
            stage_placeholder.markdown(
                badge("Researcher", stage_status["researcher"]) + " | " +
                badge("Analyst", stage_status["analyst"]) + " | " +
                badge("Writer", stage_status["writer"]),
                unsafe_allow_html=True
            )

        render_stages()
        result_text = ""

        try:
            with requests.post(
                f"{BACKEND_URL}/start-job-stream",
                json={"topic": topic},
                stream=True,
                timeout=900
            ) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if not raw.startswith("data: "):
                        continue
                    payload = raw[6:]
                    try:
                        # Backend sends simple JSON-like strings; normalize quotes for json.loads
                        evt = json.loads(payload.replace("'", '"'))
                        kind = evt.get("kind")
                        data = evt.get("data")

                        if kind == "stage":
                            if data == "researcher:start":
                                stage_status["researcher"] = "running"
                            elif data == "researcher:done":
                                stage_status["researcher"] = "done"
                            elif data == "analyst:start":
                                stage_status["analyst"] = "running"
                            elif data == "analyst:done":
                                stage_status["analyst"] = "done"
                            elif data == "writer:start":
                                stage_status["writer"] = "running"
                            elif data == "writer:done":
                                stage_status["writer"] = "done"
                            render_stages()

                        elif kind == "token":
                            result_text += str(data)
                            output_placeholder.markdown(result_text)

                        elif kind == "final":
                            rid = (data or {}).get("report_id")
                            title = (data or {}).get("title") or topic
                            st.success(f"Saved report: {title} ({rid})")
                    except Exception:
                        # Ignore parse errors from any odd lines
                        continue
            toast_success("Streaming completed")
        except Exception as e:
            toast_error(f"Stream failed: {e}")

        if result_text:
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download report.md", data=result_text, file_name="report.md", mime="text/markdown")
            with c2:
                st.code(result_text[:4000], language="markdown")

# ========== Search Tab ==========
with tabs[1]:
    st.subheader("Semantic search in saved reports")
    query = st.text_input("Search query", placeholder="e.g., transformer architectures, quantization latency, vector databases comparison")
    if st.button("Search"):
        try:
            r = requests.post(f"{BACKEND_URL}/search-reports", json={"query": query}, timeout=30)
            r.raise_for_status()
            results = r.json()
            if not results:
                st.info("No results yet. Generate some reports first.")
            for res in results:
                title = res.get("title") or "(untitled)"
                with st.expander(f"{title} — ID: {res['id']} | Score: {res['score']:.2f}", expanded=False):
                    st.markdown(res["text"])
        except Exception as e:
            toast_error(f"Search failed: {e}")
