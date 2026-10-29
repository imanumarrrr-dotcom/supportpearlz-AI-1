"""SupportPearlz AI - Streamlit console.   streamlit run app.py"""
from __future__ import annotations

import html
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from src.chains.rag_chain import answer_question
from src.config import env_api_key, settings
from src.ingestion.build_index import build_index, sync_index
from src.providers import cosine, make_embeddings
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import IndexMismatchError, IndexNotFoundError, index_exists, load_vector_store, read_manifest
from src.utils.logging_setup import append_jsonl, get_logger, read_jsonl

logger = get_logger()
st.set_page_config(page_title="SupportPearlz AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ themes (animated via registered CSS properties)
THEMES = {
    "Aurora":   dict(bg="#768383", c1="#52e5ff", c2="#9b7bff", c3="#ff70c8"),
    "Sunset":   dict(bg="#92227c", c1="#ffb347", c2="#ff5e7e", c3="#c86bff"),
    "Emerald":  dict(bg="#06130f", c1="#34f5b4", c2="#2bc4ff", c3="#b6ff6b"),
    "Midnight": dict(bg="#182c67", c1="#7aa2ff", c2="#a78bfa", c3="#f0abfc"),
}
PAGES = ["💬 Chat Console", "📚 Knowledge Base", "🔬 Retrieval Lab", "🧪 Evaluation", "📊 Analytics & Logs", "⚙️ Settings"]
STARTERS = ["How long is the warranty on the AquaPearl 500 Pro?", "My purifier beeps three times and the light flashes red.",
            "Can I return it?", "Do you offer student discounts?"]

for key, default in {"api_key": "", "authenticated": False, "messages": [], "store": None, "index_ready": False,
                     "last_report": [], "theme_name": "Aurora", "session_id": uuid.uuid4().hex[:8], "fb_logged": set()}.items():
    st.session_state.setdefault(key, default)


def inject_css(name: str) -> None:
    t = THEMES[name]
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
@property --bg {{syntax:'<color>';inherits:true;initial-value:#070b16;}}
@property --c1 {{syntax:'<color>';inherits:true;initial-value:#52e5ff;}}
@property --c2 {{syntax:'<color>';inherits:true;initial-value:#9b7bff;}}
@property --c3 {{syntax:'<color>';inherits:true;initial-value:#ff70c8;}}
:root{{--bg:{t['bg']};--c1:{t['c1']};--c2:{t['c2']};--c3:{t['c3']};--text:#edf2ff;--muted:#9aa8c7;
--panel:rgba(18,27,48,.72);--line:rgba(145,170,255,.20);
transition:--bg .9s ease,--c1 .9s ease,--c2 .9s ease,--c3 .9s ease;}}
html,body,.stApp{{font-family:'DM Sans',sans-serif;color:var(--text);}}
.stApp{{background:radial-gradient(ellipse at 12% 0%,color-mix(in srgb,var(--c2) 26%,transparent),transparent 42%),
radial-gradient(ellipse at 92% 8%,color-mix(in srgb,var(--c1) 20%,transparent),transparent 40%),
radial-gradient(ellipse at 50% 105%,color-mix(in srgb,var(--c3) 14%,transparent),transparent 45%),var(--bg);
background-size:140% 140%,140% 140%,140% 140%,auto;background-attachment:fixed;animation:drift 22s ease-in-out infinite alternate;}}
@keyframes drift{{0%{{background-position:0% 0%,100% 0%,50% 100%,0 0}}100%{{background-position:12% 8%,88% 10%,45% 92%,0 0}}}}
h1,h2,h3,h4{{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em;}}
header[data-testid="stHeader"],[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"],
[data-testid="stAppViewContainer"],[data-testid="stMain"]{{background:transparent!important;}}
.block-container{{padding-top:3.4rem;max-width:1400px;}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,color-mix(in srgb,var(--c2) 14%,#0b1020),#080c18)!important;border-right:1px solid var(--line);}}
[data-testid="stSidebar"] *,[data-testid="stSidebar"] label p{{color:var(--text)!important;}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{{color:var(--muted)!important;}}
div[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea,.stNumberInput input{{background:rgba(14,22,42,.9)!important;color:var(--text)!important;border-color:var(--line)!important;border-radius:12px!important;}}
div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{{background:#0f1730!important;color:var(--text)!important;}}
[data-testid="stChatInput"]{{background:var(--panel)!important;border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(10px);}}
[data-testid="stChatInput"] textarea{{color:var(--text)!important;}}
div[data-testid="stChatMessage"]{{border:1px solid var(--line);border-radius:16px;background:var(--panel);padding:.9rem;animation:rise .45s ease both;}}
.hero{{padding:1.5rem 1.7rem;border:1px solid var(--line);border-radius:24px;margin-bottom:1rem;position:relative;overflow:hidden;animation:rise .7s ease both;
background:linear-gradient(125deg,color-mix(in srgb,var(--c2) 22%,rgba(18,26,48,.9)),rgba(14,20,38,.7));box-shadow:0 18px 70px rgba(0,0,0,.25);}}
.hero:after{{content:'';position:absolute;width:300px;height:300px;right:-80px;top:-140px;border-radius:50%;
background:radial-gradient(circle,color-mix(in srgb,var(--c1) 35%,transparent),transparent 68%);animation:float 7s ease-in-out infinite;}}
.hero h1{{font-size:2.4rem;line-height:1.1;margin:.3rem 0 .6rem;color:#fff;}}
.eyebrow{{color:var(--c1);font-size:.74rem;font-weight:700;text-transform:uppercase;letter-spacing:.17em;}}
.gradient{{background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3),var(--c1));background-size:250% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 6s linear infinite;}}
@keyframes shine{{to{{background-position:250% 0}}}}
.muted{{color:var(--muted);}}
.pill{{display:inline-block;padding:.22rem .7rem;border:1px solid color-mix(in srgb,var(--c1) 40%,transparent);border-radius:100px;color:var(--c1);font-size:.75rem;background:color-mix(in srgb,var(--c1) 8%,transparent);margin-right:.35rem;}}
.pill.warn{{color:#ffd166;border-color:#ffd16655;background:#ffd16614;}} .pill.bad{{color:#ff7a90;border-color:#ff7a9055;background:#ff7a9014;}} .pill.ok{{color:#5dffb0;border-color:#5dffb055;background:#5dffb014;}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:#5dffb0;margin-right:6px;animation:pulse 1.8s infinite;}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 #5dffb088}}100%{{box-shadow:0 0 0 9px #5dffb000}}}}
.metric{{border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem;background:var(--panel);transition:transform .25s,border-color .25s;animation:rise .65s ease both;}}
.metric:hover{{transform:translateY(-4px);border-color:var(--c1);}}
.metric .label{{font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;}}
.metric .value{{font-family:'Space Grotesk';font-size:1.5rem;font-weight:700;margin-top:.3rem;}}
.source{{padding:.8rem 1rem;border:1px solid var(--line);border-radius:14px;background:rgba(16,25,44,.7);margin:.4rem 0;transition:all .25s;}}
.source:hover{{border-color:var(--c2);transform:translateX(3px);}}
.stButton>button,.stDownloadButton>button{{border:1px solid color-mix(in srgb,var(--c1) 45%,transparent);border-radius:12px;color:#fff;font-weight:600;
background:linear-gradient(100deg,color-mix(in srgb,var(--c1) 30%,#10182b),color-mix(in srgb,var(--c2) 38%,#10182b));transition:all .25s;}}
.stButton>button:hover,.stDownloadButton>button:hover{{border-color:var(--c1);box-shadow:0 0 26px color-mix(in srgb,var(--c1) 30%,transparent);transform:translateY(-1px);color:#fff;}}
@keyframes rise{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(24px)}}}}
@media(prefers-reduced-motion:reduce){{*,*:before,*:after{{animation:none!important;transition:none!important}}}}
</style>""", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, accent: str, sub: str, pills: str = "") -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title} <span class="gradient">{accent}</span></h1>'
                f'<p class="muted">{sub}</p>{pills}</div>', unsafe_allow_html=True)


def metric(label: str, value, note: str = "") -> None:
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="muted" style="font-size:.78rem">{note}</div></div>', unsafe_allow_html=True)


def theme_picker(key: str) -> None:
    def _apply():
        st.session_state.theme_name = st.session_state[key]
    st.selectbox("🎨 Theme", list(THEMES), index=list(THEMES).index(st.session_state.theme_name), key=key, on_change=_apply)


def validate_key(key: str) -> tuple[bool, str]:
    try:
        OpenAI(api_key=key, timeout=15, max_retries=0).models.list()
        return True, "API key verified."
    except AuthenticationError:
        return False, "Authentication failed. Check that the API key is correct and active."
    except RateLimitError:
        return False, "Provider returned a rate-limit/billing response. Check quota and billing."
    except APIConnectionError:
        return False, "Could not reach the provider. Check your internet connection and try again."
    except Exception as exc:  # noqa: BLE001
        return False, f"Validation failed: {type(exc).__name__}"


def try_autoload() -> None:
    """After login: load the persisted index automatically if one exists (never re-embeds)."""
    if st.session_state.index_ready or not index_exists():
        return
    try:
        st.session_state.store = load_vector_store(st.session_state.api_key, settings.vector_store_path, settings.embedding_model)
        st.session_state.index_ready = True
    except (IndexNotFoundError, IndexMismatchError) as exc:
        st.session_state.index_error = str(exc)
    except Exception:  # noqa: BLE001
        logger.exception("Auto-load of index failed")


def render_sources(sources: list[str], retrieved: dict) -> None:
    for label in sources:
        s = retrieved.get(label.strip("[]"), {})
        st.markdown(f'<div class="source"><b>{html.escape(label)}</b> · {html.escape(str(s.get("source", "?")))} · '
                    f'<i>{html.escape(str(s.get("location", "")))}</i><br><span class="muted">relevance {s.get("score", "n/a")} · '
                    f'{html.escape(str(s.get("doc_type", "")))} · v{html.escape(str(s.get("version", "")))}</span><br>'
                    f'<small>{html.escape(str(s.get("snippet", "")))}</small></div>', unsafe_allow_html=True)


def render_meta(out: dict) -> None:
    conf = {"high": "ok", "partial": "warn", "none": "bad"}[out["confidence"]]
    flags = " ".join(f'<span class="pill warn">{html.escape(f)}</span>' for f in out.get("flags", []) if f != "model_refusal")
    st.markdown(f'<span class="pill {conf}">confidence: {out["confidence"]}</span><span class="pill">{out["latency_ms"]} ms</span>'
                f'<span class="pill">best score {out["best_score"]}</span>{flags}<br><span class="muted" style="font-size:.78rem">'
                f'rewritten query: {html.escape(out["rewritten_query"] or "-")}</span>', unsafe_allow_html=True)


def transcript_md() -> str:
    lines = [f"# SupportPearlz chat (session {st.session_state.session_id})", ""]
    for m in st.session_state.messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}")
        for s in m.get("sources", []):
            r = m["retrieved"].get(s.strip("[]"), {})
            lines.append(f"  - {s} {r.get('source')} - {r.get('location')}")
        lines.append("")
    return "\n".join(lines)


inject_css(st.session_state.theme_name)

# ================================================================== 1) API-key gate
if not st.session_state.authenticated:
    hero("PRIVATE KNOWLEDGE • GROUNDED ANSWERS", "SupportPearlz", "AI",
         "Your intelligent customer-support workspace. Authenticate your model provider before entering the console.",
         '<span class="pill">✦ LangChain RAG</span><span class="pill">Source-aware</span><span class="pill">Session-secure</span>')
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("### 🔐 Connect your AI provider")
        st.caption("Your key is used to verify the connection and to run chat/embedding requests. It lives only in this browser session: it is never written to a file or log.")
        key = st.text_input("OpenAI API key", type="password", placeholder="sk-…", help="A key with API access and available billing/quota.")
        c1, c2 = st.columns([2, 1])
        with c1:
            go = st.button("Verify key & open console", use_container_width=True, type="primary")
        with c2:
            use_env = st.button("Use .env key", use_container_width=True, disabled=not env_api_key(), help="Uses OPENAI_API_KEY from your .env/shell.")
        if go or use_env:
            candidate = (env_api_key() if use_env else key).strip()
            if not candidate:
                st.error("Enter an API key first.")
            else:
                with st.spinner("Verifying provider credentials…"):
                    ok, message = validate_key(candidate)
                if ok:
                    st.session_state.api_key, st.session_state.authenticated = candidate, True
                    st.rerun()
                else:
                    st.error(message)
        theme_picker("theme_login")
        st.markdown('<p class="muted" style="font-size:.8rem">Security: avoid shared/public computers and use “Disconnect” when finished.</p>', unsafe_allow_html=True)
    st.stop()

try_autoload()

# ================================================================== sidebar
with st.sidebar:
    st.markdown("## ✦ SupportPearlz")
    st.caption("AI support operations console")
    st.markdown('<span class="pill ok"><span class="dot"></span>Provider connected</span>', unsafe_allow_html=True)
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    theme_picker("theme_side")
    st.divider()
    st.markdown("**Retrieval controls**")
    k = st.slider("Top-k chunks", 1, 10, settings.retrieval_k)
    threshold = st.slider("Relevance threshold", 0.0, 1.0, float(settings.score_threshold), 0.05, help="Chunks scoring below this cosine similarity are ignored; if none remain the assistant refuses without calling the model.")
    strategy = st.radio("Strategy", ["similarity", "mmr"], horizontal=True, help="MMR trades some relevance for diversity.")
    doc_filter = st.selectbox("Knowledge filter", ["All types", "policy", "manual", "guide", "faq", "pricing", "general"])
    st.caption(f"Chat model `{settings.model}` · temperature {settings.temperature}")
    if st.button("↻ Reset conversation", use_container_width=True):
        st.session_state.messages, st.session_state.session_id = [], uuid.uuid4().hex[:8]
        st.rerun()
    if st.button("Disconnect API key", use_container_width=True):
        for k_ in ("api_key", "store"):
            st.session_state[k_] = "" if k_ == "api_key" else None
        st.session_state.update(authenticated=False, index_ready=False, messages=[])
        st.rerun()

manifest = read_manifest() or {}

# ================================================================== Chat Console
if page == PAGES[0]:
    hero("CUSTOMER EXPERIENCE INTELLIGENCE", "Ask better.", "Resolve faster.",
         "Answers grounded in your knowledge base, with evidence you can inspect. Unknown facts stay unknown.",
         '<span class="pill ok"><span class="dot"></span>Secure session</span><span class="pill">↗ Citation-aware RAG</span><span class="pill">Refuses when unsure</span>')
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("Session turns", len(st.session_state.messages) // 2, f"session {st.session_state.session_id}")
    with c2: metric("Index status", "Ready" if st.session_state.index_ready else "Not loaded", f"{manifest.get('chunks', 0)} chunks · {len(manifest.get('sources', {}))} docs")
    with c3: metric("Retrieval", f"k={k}", f"{strategy} · gate {threshold:.2f}")
    with c4: metric("Embedding", manifest.get("dimension", "—"), manifest.get("embedding_model", settings.embedding_model))
    if not st.session_state.index_ready:
        st.warning(st.session_state.get("index_error") or "Knowledge index is not loaded yet. Open **Knowledge Base** and build or load the index.")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                if m.get("sources"):
                    with st.expander(f"Evidence & citations ({len(m['sources'])})"):
                        render_sources(m["sources"], m["retrieved"])
                elif m["meta"].get("refused"):
                    st.caption("No supporting source was found in the knowledge base.")
                render_meta(m["meta"])
                fb = st.feedback("thumbs", key=f"fb_{st.session_state.session_id}_{i}")
                if fb is not None and (st.session_state.session_id, i) not in st.session_state.fb_logged:
                    st.session_state.fb_logged.add((st.session_state.session_id, i))
                    append_jsonl(settings.feedback_path, {"session_id": st.session_state.session_id, "turn": i, "thumbs_up": bool(fb),
                                                         "query": m["meta"]["original_query"], "answer": m["content"][:500]})

    if not st.session_state.messages:
        st.markdown("##### Try a question")
        cols = st.columns(len(STARTERS))
        for col, s in zip(cols, STARTERS):
            if col.button(s, key=f"st_{s}", use_container_width=True):
                st.session_state.queued = s
                st.rerun()
    else:
        st.download_button("⬇ Export transcript", transcript_md(), file_name=f"supportpearlz_{st.session_state.session_id}.md", mime="text/markdown")

    question = st.chat_input("Ask about warranty, returns, shipping, troubleshooting…") or st.session_state.pop("queued", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        if not st.session_state.index_ready or st.session_state.store is None:
            st.session_state.messages += [{"role": "user", "content": question},
                                          {"role": "assistant", "content": "The knowledge index is not loaded. Open **Knowledge Base**, then build or load the index first."}]
            st.rerun()
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and drafting a grounded answer…"):
                out = answer_question(question, st.session_state.messages, st.session_state.store, st.session_state.api_key,
                                      settings.model, settings.temperature, k, threshold, None if doc_filter == "All types" else doc_filter,
                                      strategy, st.session_state.session_id)
        st.session_state.messages += [{"role": "user", "content": question},
                                      {"role": "assistant", "content": out["answer"], "sources": out["sources"], "retrieved": out["retrieved"], "meta": out}]
        st.rerun()

# ================================================================== Knowledge Base
elif page == PAGES[1]:
    hero("INGESTION PIPELINE", "Knowledge", "Base", "Ingestion is separate from chat: build once, persist to disk, then query the index.")
    uploaded = st.file_uploader("Add knowledge documents (PDF, DOCX, Markdown, TXT, CSV)", type=["pdf", "docx", "md", "txt", "csv"], accept_multiple_files=True)
    if uploaded and st.button("Save uploaded documents"):
        target = Path(settings.knowledge_base_path)
        target.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (target / Path(f.name).name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} document(s). Use **Sync changes** (fast) or **Rebuild** to index them.")
    st.markdown("#### Index configuration")
    a, b, c = st.columns(3)
    with a: chunk_size = st.number_input("Chunk size (characters)", 200, 2500, settings.chunk_size, step=100)
    with b: overlap = st.number_input("Chunk overlap", 0, 1000, min(settings.chunk_overlap, settings.chunk_size - 1), step=20)
    with c: st.metric("Embedding model", settings.embedding_model)
    docs_path = Path(settings.knowledge_base_path)
    available = sorted(p for p in docs_path.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".md", ".txt", ".csv"}) if docs_path.exists() else []
    st.info(f"{len(available)} supported file(s) in `{docs_path.name}/`: " + ", ".join(p.name for p in available))
    b1, b2, b3 = st.columns(3)
    build = b1.button("⚡ Build / Rebuild index", use_container_width=True, type="primary", help="Wipes the collection and re-indexes everything (embeddings are cached).")
    sync = b2.button("🔄 Sync changes only", use_container_width=True, help="Delete-by-source + reinsert only new/changed files.")
    load = b3.button("📂 Load existing index", use_container_width=True)
    try:
        if build or sync:
            if overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
            else:
                with st.spinner("Loading files, splitting, embedding and persisting Chroma…"):
                    if build:
                        store, report, info = build_index(st.session_state.api_key, int(chunk_size), int(overlap))
                    else:
                        store, report, info = sync_index(st.session_state.api_key)
                st.session_state.update(store=store, index_ready=True, last_report=report)
                st.success(f"Done. {json.dumps({k_: v for k_, v in info.items()})}")
        if load:
            st.session_state.store = load_vector_store(st.session_state.api_key, settings.vector_store_path, settings.embedding_model)
            st.session_state.index_ready = True
            st.success("Existing persistent index loaded from disk. Nothing was re-embedded.")
    except (IndexNotFoundError, IndexMismatchError) as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Index operation failed")
        st.error("The index operation failed safely (details in the logs). Check the API key, quota and network.")
    manifest = read_manifest() or {}
    if manifest:
        st.markdown("#### Index manifest")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunks", manifest.get("chunks")); m2.metric("Documents", len(manifest.get("sources", {})))
        m3.metric("Vector dimension", manifest.get("dimension")); m4.metric("Built", str(manifest.get("built_at", ""))[:16].replace("T", " "))
    if st.session_state.last_report:
        st.markdown("#### Last ingestion report")
        st.dataframe(pd.DataFrame(st.session_state.last_report), use_container_width=True)
    if st.session_state.index_ready and st.session_state.store is not None:
        st.markdown("#### Chunk explorer")
        src = st.selectbox("Document", sorted(manifest.get("sources", {})))
        got = st.session_state.store.get(where={"source": src}, include=["documents", "metadatas"])
        for d, md in sorted(zip(got["documents"], got["metadatas"]), key=lambda x: x[1].get("chunk_index", 0))[:30]:
            st.markdown(f'<div class="source"><b>{html.escape(md.get("chunk_id", ""))}</b> · {html.escape(str(md.get("location", "")))} · {html.escape(str(md.get("doc_type", "")))}'
                        f'<br><small>{html.escape(d[:400])}</small></div>', unsafe_allow_html=True)

# ================================================================== Retrieval Lab
elif page == PAGES[2]:
    hero("RETRIEVAL DEBUGGING", "Retrieval", "Lab", "Inspect scores without involving the LLM. Calibrate your relevance gate on your own data.")
    if not st.session_state.index_ready:
        st.warning("Load or build the index first.")
        st.stop()
    from evaluation.retrieval_eval import calibrate_threshold, chunk_experiment, compare_strategies, load_cases
    store = st.session_state.store
    q = st.text_input("Search query", "how long is the warranty")
    if q:
        res = retrieve(store, q, k, 0.0, strategy, None if doc_filter == "All types" else doc_filter, fetch_k=settings.fetch_k)
        st.dataframe(pd.DataFrame([{"rank": h.label, "score": h.score, "passes gate": h.score >= threshold, "source": h.source, "location": h.location,
                                    "preview": h.text[:200].replace("\n", " ")} for h in res.all_hits]), use_container_width=True)
        st.caption(f"Gate ({threshold:.2f}): {'PASS' if any(h.score >= threshold for h in res.all_hits) else 'REFUSE - no chunk reaches the threshold'} · best score {res.best_score:.3f}")
    t1, t2, t3, t4 = st.tabs(["Strategy comparison (hit@k)", "Threshold calibration", "Embedding diagnostic", "Chunking experiment"])
    with t1:
        if st.button("Run strategy comparison"):
            st.dataframe(pd.DataFrame(compare_strategies(store, load_cases())), use_container_width=True)
    with t2:
        if st.button("Calibrate threshold"):
            cal = calibrate_threshold(store, load_cases())
            st.json(cal)
            if cal["unanswerable_max"] is not None and cal["answerable_min"] is not None:
                gap = cal["answerable_min"] - cal["unanswerable_max"]
                st.info(f"Answerable min {cal['answerable_min']:.3f} vs unanswerable max {cal['unanswerable_max']:.3f}. "
                        + (f"Suggested gate ≈ {(cal['answerable_min'] + cal['unanswerable_max']) / 2:.2f}." if gap > 0 else "Distributions overlap - rely on the prompt refusal rule as a second layer."))
    with t3:
        s1 = st.text_input("Sentence A", "My machine keeps beeping three times")
        s2 = st.text_input("Sentence B (paraphrase)", "Error E-03: intermittent audible alarm indicating filter saturation")
        s3 = st.text_input("Sentence C (unrelated)", "The capital of France is Paris")
        if st.button("Compute cosine similarities"):
            v = make_embeddings(st.session_state.api_key).embed_documents([s1, s2, s3])
            st.table(pd.DataFrame({"pair": ["A–B (paraphrase)", "A–C (unrelated)", "B–C (unrelated)"], "cosine": [round(cosine(v[0], v[1]), 3), round(cosine(v[0], v[2]), 3), round(cosine(v[1], v[2]), 3)]}))
            st.caption(f"Model {settings.embedding_model} · dimension {len(v[0])}")
    with t4:
        st.caption("Builds in-memory indexes for 7 configurations (embeddings are cached, so repeats are cheap) and reports hit@4.")
        if st.button("Run chunking experiment"):
            from src.ingestion.loaders import load_documents
            with st.spinner("Embedding and scoring configurations…"):
                st.dataframe(pd.DataFrame(chunk_experiment(load_documents(settings.knowledge_base_path)[0], st.session_state.api_key, load_cases())), use_container_width=True)

# ================================================================== Evaluation
elif page == PAGES[3]:
    hero("QUALITY & MEASUREMENT", "Evaluation", "Lab", "Scores come from measured outcomes. Automated scoring is a heuristic - verify non-pass rows manually.")
    from evaluation.retrieval_eval import load_cases
    cases = load_cases()
    st.dataframe(pd.DataFrame(cases).groupby("category").size().rename("cases").reset_index(), use_container_width=True)
    st.caption(f"{len(cases)} test cases (easy, multi-step, follow-up, unanswerable, ambiguous, adversarial, robustness).")
    results_path = Path("evaluation/results/eval_results.json")
    if st.session_state.index_ready and st.button("▶ Run full evaluation", type="primary"):
        from evaluation.run_eval import run_evaluation
        bar = st.progress(0.0, text="Running…")
        out = run_evaluation(st.session_state.store, st.session_state.api_key, k, threshold, strategy,
                             progress=lambda i, n, r: bar.progress(i / n, text=f"{r['id']} → {r['verdict']}"))
        bar.empty()
    elif not st.session_state.index_ready:
        st.warning("Load or build the index first.")
    if results_path.exists():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        agg = data["aggregate"]
        cols = st.columns(5)
        for col, (label, val) in zip(cols, [("Pass / partial / fail", f"{agg['pass']}/{agg['partial']}/{agg['fail']}"), ("Correctness", agg["correctness_rate"]),
                                            ("Groundedness", agg["groundedness_rate"]), ("Refusal rate (unanswerable)", agg["refusal_rate_on_unanswerable"]),
                                            ("Hallucination flags", agg["hallucination_count_heuristic"])]):
            col.metric(label, val)
        st.dataframe(pd.DataFrame(data["results"])[["id", "category", "verdict", "correctness", "groundedness", "citation_accuracy", "suspected_stage", "rewritten_query", "answer"]], use_container_width=True)
        st.download_button("⬇ Download results JSON", results_path.read_bytes(), file_name="eval_results.json", mime="application/json")
    if st.session_state.index_ready:
        with st.expander("Prompt experiment: weak prompt (v0) vs final prompt (v3)"):
            if st.button("Run prompt experiment"):
                from evaluation.prompt_experiment import run as run_prompt_exp
                with st.spinner("Calling the model…"):
                    st.dataframe(pd.DataFrame(run_prompt_exp(st.session_state.store, st.session_state.api_key, k, threshold)), use_container_width=True)

# ================================================================== Analytics & Logs
elif page == PAGES[4]:
    hero("OBSERVABILITY", "Analytics &", "Logs", "Live metrics from the per-turn log: latency, refusals, retrieval quality and user feedback.")
    rows = read_jsonl(settings.turn_log_path)
    if rows:
        df = pd.DataFrame(rows)
        a, b, c, d = st.columns(4)
        a.metric("Questions", len(df)); b.metric("Refusal rate", f"{df['refused'].mean():.0%}")
        c.metric("Median latency", f"{int(df['latency_ms'].median())} ms"); d.metric("p95 latency", f"{int(df['latency_ms'].quantile(.95))} ms")
        st.markdown("##### Latency per turn (ms)")
        st.line_chart(df["latency_ms"], height=200)
        st.markdown("##### Best retrieval score per turn")
        st.bar_chart(df["best_score"], height=200)
        st.dataframe(df.tail(50).iloc[::-1], use_container_width=True)
        gaps = df[df["refused"] & df["error"].isna()]["query"].tolist()
        if gaps:
            st.markdown("##### Knowledge gaps (questions that were refused)")
            st.write(pd.Series(gaps).value_counts().rename("times asked"))
    else:
        st.info("No turns logged yet. Ask a question in the Chat Console.")
    fb = read_jsonl(settings.feedback_path)
    if fb:
        up = sum(f["thumbs_up"] for f in fb)
        st.metric("User feedback 👍", f"{up}/{len(fb)}")
    log_path = Path(settings.log_file)
    if log_path.exists():
        with st.expander("Application log (last 80 lines)"):
            st.code("\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]), language="text")

# ================================================================== Settings
else:
    hero("RUNTIME CONFIGURATION", "Workspace", "Settings", "Everything below is environment-driven (.env). Change values there and restart Streamlit.")
    st.json({k_: v for k_, v in settings.__dict__.items()} if hasattr(settings, "__dict__") else str(settings))
    st.warning("The API key is held only in this session's memory. It is never displayed, written to disk or logged.")
    st.markdown("#### Safety boundaries")
    st.write("- This assistant drafts answers; it does not send customer messages or approve refunds, credits or exceptions.")
    st.write("- Safety risks, property-damage claims, exceptions and ambiguous commercial commitments require human review.")
    st.write("- Retrieved document text and user messages are treated as data, never as instructions.")