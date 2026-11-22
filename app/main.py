"""Customer Support Ticket Router - Streamlit application.

Run:  streamlit run app/main.py
"""
from __future__ import annotations

import html
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.components.analytics import clear_log, load_log, log_result  # noqa: E402
from app.components.examples import EXAMPLES  # noqa: E402
from src import config as C  # noqa: E402
from src.pipeline import MODEL_ORDER, PRIMARY_MODEL, TicketRouter  # noqa: E402

st.set_page_config(page_title="Ticket Router", page_icon="🎫", layout="wide")
st.markdown(f"<style>{(ROOT / 'app' / 'styles' / 'style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the four models (CPU)...")
def get_router() -> TicketRouter:
    return TicketRouter(MODEL_ORDER)


def esc(x) -> str:
    return html.escape(str(x))


# ----------------------------------------------------------------------------- state
st.session_state.setdefault("message", "")
st.session_state.setdefault("result", None)


def set_example(text: str) -> None:
    st.session_state.message = text
    st.session_state.result = None


def clear() -> None:
    st.session_state.message = ""
    st.session_state.result = None


# ----------------------------------------------------------------------------- header
st.markdown(
    "<div class='hero'><h1>🎫 Customer Support Ticket Router</h1>"
    "<p>Turkish / English support messages → intent → department. Routing decision made by the "
    "<b>distilled INT8-quantized Transformer</b> running on CPU; three other models shown side by side.</p></div>",
    unsafe_allow_html=True)

router = get_router()

with st.sidebar:
    st.subheader("System")
    st.markdown(f"<span class='badge int8'>Primary: {esc(C.MODEL_NAMES[PRIMARY_MODEL])}</span>", unsafe_allow_html=True)
    st.caption("All inference runs on CPU (torch threads = 4).")
    st.markdown("**Confidence thresholds** (tuned on the validation split)")
    st.dataframe(pd.DataFrame([{"model": C.MODEL_NAMES[n], "threshold": router.threshold(n)} for n in MODEL_ORDER]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Departments**")
    for d in sorted(set(C.INTENT_TO_DEPARTMENT.values())):
        st.caption(f"• {d}")

tab_classify, tab_analytics, tab_results, tab_arch = st.tabs(
    ["Classify", "Analytics", "Benchmark results", "Architecture"])

# ----------------------------------------------------------------------------- classify tab
with tab_classify:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.text_area("Customer message (Türkçe / English)", key="message", height=130,
                     placeholder="ör. bulaşık makinem su boşaltmıyor, model DW-6534")
        b1, b2 = st.columns(2)
        classify = b1.button("Classify", type="primary", use_container_width=True)
        b2.button("Clear", on_click=clear, use_container_width=True)
    with right:
        st.markdown("**Example messages**")
        for group, msgs in EXAMPLES.items():
            with st.expander(group, expanded=False):
                for i, m in enumerate(msgs):
                    st.button(m, key=f"ex-{group}-{i}", on_click=set_example, args=(m,), use_container_width=True)

    if classify:
        text = st.session_state.message
        if not text.strip():
            st.warning("Please type a message first.")
        else:
            res = router.route(text).to_dict()
            st.session_state.result = res
            log_result(res)

    res = st.session_state.result
    if res:
        lang = res["language"]
        badge = {"tr": "<span class='badge tr'>🇹🇷 Türkçe</span>", "en": "<span class='badge en'>🇬🇧 English</span>"}.get(
            lang, "<span class='badge'>unknown</span>")
        reason = res["routing_reason"]
        if reason == "rule_based_small_talk" or res["final_department"] == C.SMALL_TALK:
            cls, title = "talk", "💬 Small talk – answered automatically"
        elif reason in ("low_confidence", "out_of_scope"):
            cls = "human"
            title = ("🧑‍💼 Routed to HUMAN AGENT – out of scope" if reason == "out_of_scope"
                     else "🧑‍💼 Routed to HUMAN AGENT – low confidence")
        else:
            cls, title = "", "✅ Routed automatically"
        conf = res["confidence"]
        conf_txt = f"{conf:.1%}" if conf is not None else "—"
        st.markdown(
            f"<div class='route {cls}'>{badge}<div>{title}</div>"
            f"<div class='dept'>{esc(res['final_department'])}</div>"
            f"<div class='muted'>intent: <b>{esc(res['final_intent'])}</b> · confidence: <b>{conf_txt}</b>"
            f" · decided by {esc(C.MODEL_NAMES[PRIMARY_MODEL])}</div></div>", unsafe_allow_html=True)
        if res["small_talk_reply"]:
            with st.chat_message("assistant"):
                st.write(res["small_talk_reply"])

        q = res["models"][PRIMARY_MODEL]
        corr = ", ".join(f"{a} → {b}" for a, b in res["corrections"]) or "—"
        st.markdown(
            "<div class='card'><h4>Classification details</h4><div class='kv'>"
            f"<div>Original message</div><div>{esc(res['original'])}</div>"
            f"<div>Normalized message</div><div><code>{esc(res['normalized'])}</code></div>"
            f"<div>Language</div><div>{esc(lang)}</div>"
            f"<div>Product model code</div><div><b>{esc(', '.join(res['model_codes']) or '—')}</b></div>"
            f"<div>Product category</div><div>{esc(res['product_category'] or '—')}</div>"
            f"<div>Spelling corrections</div><div>{esc(corr)}</div>"
            f"<div>Predicted intent (raw)</div><div>{esc(q['intent'])} ({q['confidence']:.1%})</div>"
            f"<div>Department</div><div>{esc(res['final_department'])}</div>"
            f"<div>Routing decision</div><div>{esc(reason)}</div>"
            "</div></div>", unsafe_allow_html=True)

        st.markdown("#### Four-model comparison")
        cols = st.columns(4)
        for col, name in zip(cols, MODEL_ORDER):
            m = res["models"][name]
            primary = " primary" if name == PRIMARY_MODEL else ""
            tag = "<span class='badge int8'>INT8 · used for routing</span>" if name == PRIMARY_MODEL else ""
            human = " 🧑‍💼" if m["routed_to_human"] else ""
            top3 = "<br>".join(f"{esc(l)}: {p:.2f}" for l, p in m["top3"])
            col.markdown(
                f"<div class='card{primary}'><h4>{esc(m['display_name'])}</h4>{tag}"
                f"<div class='big'>{esc(m['final_intent'])}{human}</div>"
                f"<div class='muted'>{esc(m['department'])}</div>"
                f"<div style='margin-top:6px'>confidence <b>{m['confidence']:.1%}</b></div>"
                f"<div class='conf-bar'><span style='width:{m['confidence'] * 100:.0f}%'></span></div>"
                f"<div style='margin-top:6px'>⏱ <b>{m['latency_ms']:.1f} ms</b></div>"
                f"<div class='muted' style='margin-top:6px'>top-3:<br>{top3}</div></div>",
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- analytics tab
with tab_analytics:
    log = load_log()
    st.caption("Computed from messages actually processed by this app (artifacts/app_logs/predictions.jsonl).")
    if log.empty:
        st.info("No messages processed yet. Classify a few messages first.")
    else:
        human = log.routing_reason.isin(["low_confidence", "out_of_scope"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Processed messages", len(log))
        k2.metric("Routed to human agent", f"{int(human.sum())} ({human.mean():.0%})")
        k3.metric("Out-of-scope", int((log.routing_reason == "out_of_scope").sum()))
        k4.metric("Low confidence", int((log.routing_reason == "low_confidence").sum()))
        c1, c2 = st.columns(2)
        c1.markdown("**Intent distribution (primary model)**")
        c1.bar_chart(log.final_intent.value_counts(), horizontal=True)
        c2.markdown("**Department distribution**")
        c2.bar_chart(log.final_department.value_counts(), horizontal=True)
        c3, c4 = st.columns(2)
        lat = pd.DataFrame({C.MODEL_NAMES[n]: [log[f"{n}_latency_ms"].mean()] for n in MODEL_ORDER
                            if f"{n}_latency_ms" in log}).T.rename(columns={0: "avg latency (ms)"})
        c3.markdown("**Average inference latency in the app (ms)**")
        c3.bar_chart(lat, horizontal=True)
        c4.markdown("**Language mix**")
        c4.bar_chart(log.language.value_counts())
        agree = {C.MODEL_NAMES[n]: float((log[f"{n}_intent"] == log[f"{PRIMARY_MODEL}_intent"]).mean())
                 for n in MODEL_ORDER if f"{n}_intent" in log}
        st.markdown("**Agreement with the primary (INT8) model on processed messages**")
        st.dataframe(pd.DataFrame([agree]), hide_index=True)
        with st.expander("Raw log"):
            st.dataframe(log, use_container_width=True)
        if st.button("Reset analytics log"):
            clear_log()
            st.rerun()

# ----------------------------------------------------------------------------- results tab
with tab_results:
    res_path = C.REPORTS / "results.csv"
    if not res_path.exists():
        st.info("Run `python scripts/evaluate_all.py` to create the benchmark results.")
    else:
        df = pd.read_csv(res_path)
        hw = json.loads((C.REPORTS / "hardware.json").read_text())
        st.markdown("Same held-out test set for all models "
                    f"(n = {int(df.n.iloc[0])}, of which {int(df.n_oos.iloc[0])} out-of-scope). "
                    f"Latency: batch 1, {int(df.warmup.iloc[0])} warm-up + {int(df.runs.iloc[0])} timed runs, "
                    f"CPU {hw['cpu']}, {hw['ram_gb']} GB RAM, {hw['benchmark_threads']} threads.")
        show = df[["display_name", "accuracy", "macro_f1", "oos_recall", "size_mb", "latency_mean_ms"]].rename(columns={
            "display_name": "Model", "accuracy": "Accuracy", "macro_f1": "Macro-F1", "oos_recall": "OOS recall",
            "size_mb": "Size (MB)", "latency_mean_ms": "Latency (ms)"})
        st.dataframe(show.style.format({"Accuracy": "{:.3f}", "Macro-F1": "{:.3f}", "OOS recall": "{:.3f}",
                                        "Size (MB)": "{:.1f}", "Latency (ms)": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        figs = sorted(C.FIGURES.glob("*.png"))
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for col, f in zip(cols, figs[i:i + 2]):
                col.image(str(f), use_container_width=True)

# ----------------------------------------------------------------------------- architecture tab
with tab_arch:
    arch = ROOT / "docs" / "architecture.png"
    if arch.exists():
        st.image(str(arch), use_container_width=True)
    else:
        st.info("docs/architecture.png not found.")
