#!/usr/bin/python3
"""
app_albert.py — ViromeChat-AI adapted for Albert API (etalab / OpenAI-compatible)

Changes vs original app.py (Ollama):
  - HTTP calls → POST https://albert.api.etalab.gouv.fr/v1/chat/completions
  - Auth header → Authorization: Bearer <ALBERT_API_KEY>
  - Response parsing → choices[0].message  (OpenAI format)
  - tool_calls arguments → JSON string, need json.loads()
  - Model listing → GET /v1/models
  - Removed Ollama-specific options (repeat_penalty, seed)
  - Added retry logic on 429 rate-limit responses
  - Robust argument parser for known vLLM/gpt-oss-120b tool_call bug
"""

import streamlit as st
import pandas as pd
import requests
import os
import json
import logging
import time
import re
from datetime import datetime

from tools import (
    wikipedia_search, pubmed_search, TOOL_LABELS,
    query_dataframe, create_visualization, create_map, TOOLS_SPEC,
)
from config import (
    TAXO_DB_PATH, HOST_DB_PATH, LOG_DIR,
    PAGE_ICON, GITHUB_URL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CONTENT,
    DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT,
)

# ── Albert API constants ──────────────────────────────────────────────────────
ALBERT_BASE_URL      = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_TIMEOUT       = 180          # seconds — large model can be slow
ALBERT_MODEL_DEFAULT = "AgentPublic/gptoss120b"  # adjust if exact name differs
REPORT_DIR           = os.path.join(LOG_DIR, "error_reports")

from logging_utils import setup_logger
logger = setup_logger(LOG_DIR)

# ==================== PMID HALLUCINATION GUARD ==================== #

import re as _re

def _strip_hallucinated_pmids(text: str, real_pmids: set) -> tuple[str, list]:
    pattern = _re.compile(r'\bPMID[:\s#]*([0-9]{5,9})\b', _re.IGNORECASE)
    removed = []
    
    def _replace(m):
        pmid = m.group(1)
        if pmid in real_pmids:
            return m.group(0)
        removed.append(pmid)
        return ""
    
    cleaned = pattern.sub(_replace, text)
    
    # Nettoyer les ponctuations orphelines
    cleaned = _re.sub(r'\(e\.g\.\s*,?\s*on\s+[^)]{0,80}\)', '', cleaned)
    cleaned = _re.sub(r'\(see\s*\)', '', cleaned)
    
    # CORRECTION ICI : ne remplacer que les espaces/tabulations multiples
    # [ \t]+ = espace ou tabulation, mais PAS \n
    cleaned = _re.sub(r'[ \t]+', ' ', cleaned)
    
    # Préserver la structure markdown : max 2 sauts de ligne consécutifs
    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned)
    
    cleaned = cleaned.strip()
    return cleaned, removed
# ==================== PASSWORD ==================== #

def check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("Access Required")
    pwd = st.text_input("Enter access code", type="password")
    if st.button("Login"):
        if pwd == st.secrets.get("ACCESS_CODE", ""):
            st.session_state.authenticated = True
            logger.info("AUTH | Login successful")
            st.rerun()
        else:
            logger.warning("AUTH | Failed login attempt")
            st.error("Invalid access code")
    st.stop()


# ==================== DATA LOADING ==================== #

@st.cache_data(show_spinner=False)
def load_dataframe(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ==================== ALBERT API HELPERS ==================== #

def _get_api_key() -> str:
    """
    Retrieve the Albert API key.
    Priority: st.secrets["ALBERT_API_KEY"]  >  env var ALBERT_API_KEY
    """
    key = st.secrets.get("ALBERT_API_KEY", "") or os.environ.get("ALBERT_API_KEY", "")
    if not key:
        st.error(
            "Albert API key not found. "
            "Add ALBERT_API_KEY to your .streamlit/secrets.toml or environment."
        )
        st.stop()
    return key


def _albert_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }


def _list_albert_models(api_key: str) -> list:
    """Return available text-generation model names from Albert /v1/models."""
    try:
        r = requests.get(
            f"{ALBERT_BASE_URL}/models",
            headers=_albert_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        # Exclude embeddings / audio / reranking models
        names = [
            m["id"] for m in data
            if m.get("object") == "model"
            and not any(kw in m["id"].lower() for kw in ("embed", "whisper", "rerank"))
        ]
        return names if names else [ALBERT_MODEL_DEFAULT]
    except Exception as e:
        logger.warning(f"MODEL_LIST_FAIL | {e} — using default model")
        return [ALBERT_MODEL_DEFAULT]


def _parse_tool_arguments(raw_args) -> dict:
    """
    Albert / vLLM may return tool_call arguments as:
      - a dict          (ideal case)
      - a JSON string   (most common case)
      - a malformed / partial JSON string  (known bug with gpt-oss-120b)

    Returns a dict in all cases.
    Falls back to {"_raw": raw_args} if everything fails.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        # Normal case: valid JSON string
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass
        # Fallback: extract key-value pairs with a simple regex
        recovered = {}
        for m in re.finditer(
            r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null)',
            raw_args,
        ):
            try:
                recovered[m.group(1)] = json.loads(m.group(2))
            except Exception:
                recovered[m.group(1)] = m.group(2)
        if recovered:
            logger.warning(f"TOOL_ARG_PARTIAL_PARSE | recovered={recovered}")
            return recovered
        logger.error(f"TOOL_ARG_PARSE_FAIL | raw={raw_args[:300]}")
        return {"_raw": raw_args}
    return {}


def _albert_chat(
    messages: list,
    tools: list,
    model: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int = 4096,   # increased: 2048 was causing truncated answers
    retry: int = 3,
) -> dict:
    """
    POST to Albert /v1/chat/completions.
    Retries up to `retry` times on HTTP 429 (rate limit).
    """
    payload = {
        "model":       model,
        "messages":    messages,
        "tools":       tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p":       top_p,
        "max_tokens":  max_tokens,
        "stream":      False,
    }

    for attempt in range(1, retry + 1):
        try:
            r = requests.post(
                f"{ALBERT_BASE_URL}/chat/completions",
                headers=_albert_headers(api_key),
                json=payload,
                timeout=ALBERT_TIMEOUT,
            )
            if r.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"RATE_LIMIT | attempt {attempt}/{retry} — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            logger.error(f"ALBERT_TIMEOUT | attempt {attempt}")
            if attempt == retry:
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"ALBERT_HTTP_ERROR | {e}")
            raise

    raise RuntimeError("Albert API: max retries exceeded")


# ==================== ERROR REPORT ==================== #

def save_error_report(question: str, answer: str, executed_codes: list, comment: str = ""):
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp  = datetime.now()
    ts_str     = timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{ts_str}.json")

    log_filename = os.path.join(LOG_DIR, f"agent_{timestamp.strftime('%Y-%m')}.log")
    related_logs = []
    if os.path.exists(log_filename):
        try:
            with open(log_filename, "r", encoding="utf-8") as f:
                related_logs = [l.rstrip() for l in f.readlines()[-200:]]
        except Exception:
            related_logs = ["[Could not read log file]"]

    report = {
        "timestamp":      timestamp.isoformat(),
        "user_comment":   comment,
        "question":       question,
        "answer":         answer,
        "executed_codes": executed_codes,
        "recent_logs":    related_logs,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.warning(f"ERROR_REPORT | Saved to {report_path} | comment={comment!r}")
    return report_path


# ==================== SOURCES DISPLAY ==================== #

def render_sources(wikipedia_urls, pubmed_urls, executed_codes):
    if not wikipedia_urls and not pubmed_urls and not executed_codes:
        return
    with st.expander("📚 Sources"):
        if wikipedia_urls:
            st.markdown("**📘 Wikipedia**")
            for url in wikipedia_urls:
                title = url.split("/")[-1].replace("_", " ")
                st.markdown(f"- [{title}]({url})")
        if pubmed_urls:
            st.markdown("**🔬 PubMed**")
            for url in pubmed_urls:
                pmid = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
                st.markdown(f"- [PMID: {pmid}]({url})")
        if executed_codes:
            st.markdown("**📊 Dataset Query & Visualization**")
            full_code = "\n\n---\n\n".join(
                f"# Code {i}\n{code}" for i, code in enumerate(executed_codes, 1)
            )
            st.code(full_code, language="python")


# ==================== ERROR REPORT BUTTON ==================== #

def render_report_button(msg_idx: int, question: str, answer: str, executed_codes: list):
    report_key = f"reported_{msg_idx}"
    dialog_key = f"show_dialog_{msg_idx}"

    if st.session_state.get(report_key):
        st.caption("⚠️ Error reported — thank you for your feedback.")
        return

    if st.button("🚩 Report an error", key=f"btn_report_{msg_idx}",
                 help="Signal a wrong or misleading answer"):
        st.session_state[dialog_key] = not st.session_state.get(dialog_key, False)

    if st.session_state.get(dialog_key):
        with st.container(border=True):
            st.markdown("**What went wrong?** *(optional)*")
            comment = st.text_area(
                "Your comment", key=f"comment_{msg_idx}",
                placeholder="e.g. Wrong species name, incorrect count, hallucinated data…",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Send report", key=f"send_report_{msg_idx}", type="primary"):
                    save_error_report(question, answer, executed_codes, comment)
                    st.session_state[report_key] = True
                    st.session_state[dialog_key] = False
                    st.success("Report saved ✅")
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_report_{msg_idx}"):
                    st.session_state[dialog_key] = False
                    st.rerun()


# ==================== AGENT LOOP ==================== #

def albert_agent_loop(
    model: str,
    api_key: str,
    user_query: str,
    df_taxo,
    df_host,
    temperature: float      = DEFAULT_TEMPERATURE,
    top_p: float            = DEFAULT_TOP_P,
    max_tool_calls: int     = DEFAULT_MAX_TOOL_CALLS,
    preview_rows: int       = DEFAULT_PREVIEW_ROWS,
    wikipedia_limit: int    = DEFAULT_WIKIPEDIA_LIMIT,
    max_tool_content: int   = DEFAULT_MAX_TOOL_CONTENT,
    status_container        = None,
):
    """
    Agentic loop using Albert API (OpenAI-compatible /v1/chat/completions).

    Key differences vs original Ollama loop
    ----------------------------------------
    | Ollama                        | Albert / OpenAI                        |
    |-------------------------------|----------------------------------------|
    | r.json()["message"]           | r.json()["choices"][0]["message"]      |
    | "tool_calls" key in msg       | finish_reason == "tool_calls"          |
    | args already a dict           | args is a JSON *string* → json.loads() |
    | repeat_penalty / seed options | not supported → removed                |
    """

    df_taxo_columns_str = ", ".join(df_taxo.columns)
    df_host_columns_str = ", ".join(df_host.columns)

    # Build column sample strings (10 non-null values per column)
    def _col_samples(df, col, n=10):
        try:
            vals = df[col].dropna().unique()[:n]
            return ", ".join(str(v) for v in vals)
        except Exception:
            return "N/A"

    taxo_samples = {c: _col_samples(df_taxo, c) for c in df_taxo.columns}
    host_samples = {c: _col_samples(df_host, c) for c in df_host.columns}

    taxo_col_detail = "\n".join(
        f"  - {c}: e.g. {taxo_samples[c]}" for c in df_taxo.columns
    )
    host_col_detail = "\n".join(
        f"  - {c}: e.g. {host_samples[c]}" for c in df_host.columns
    )

    logger.info("STARTUP | column names loaded — df_taxo: %s | df_host: %s",
                list(df_taxo.columns), list(df_host.columns))

    system_prompt = f"""
You are a scientific bioinformatics assistant specialized in virology and viral ecology.
You have access to two curated datasets and external tools. You must ground every statement in data or tool output.

═══════════════════════════════════════════════
AVAILABLE DATA
═══════════════════════════════════════════════

`df_taxo` — Taxonomy database (NCBI/SRA)
  Columns and example values:
{taxo_col_detail}

  ★ KEY COLUMN SEMANTICS:
  - ORGANISM_NAME  → the species/genus/family name (e.g. "Bos taurus", "Homo sapiens")
  - TAX_ID         → numeric NCBI taxon identifier
  - RANK           → "species", "genus", "family", etc.
  - SPECIES_NAME   → often EMPTY — do NOT use this to search for organism names

`df_host` — Virus-host occurrence database (SRA)
  Columns and example values:
{host_col_detail}

  ★ KEY COLUMN SEMANTICS:
  - TAX_ID         → host taxon ID (links to df_taxo.TAX_ID)
  - VIRAL_TAX_ID   → virus taxon ID (links to df_taxo.TAX_ID for the virus)
  - VIRAL_SPECIES  → virus name (e.g. "bovine respiratory syncytial virus")
  - SPECIES_NAME   → often EMPTY — do NOT use to search host names
  - lon / lat      → coordinates of sampling location

JOIN KEY: df_taxo.TAX_ID ↔ df_host.TAX_ID  (host side)

⚠️ Use ONLY the column names listed above. Always filter with .str.contains() case-insensitive.

CRITICAL — HOST LOOKUP (TWO-STEP MANDATORY):
df_host.SPECIES_NAME is EMPTY — never search host names there.
To find viruses infecting a named host (e.g. "Bos taurus", "cat", "human"):

  STEP 1 — Resolve host name → TAX_ID using df_taxo.ORGANISM_NAME:
    host_rows = df_taxo[df_taxo['ORGANISM_NAME'].str.contains('Bos taurus', case=False, na=False)]
    host_taxid = int(host_rows['TAX_ID'].iloc[0])

  STEP 2 — Query df_host using the numeric TAX_ID:
    result = df_host[df_host['TAX_ID'] == host_taxid][['VIRAL_SPECIES', 'VIRAL_TAX_ID']].drop_duplicates()

- ALWAYS use ORGANISM_NAME (not SPECIES_NAME) to search names in df_taxo
- SPECIES_NAME in df_taxo is almost always empty — ignore it for name lookups
- If STEP 1 returns 0 rows, try genus only: .str.contains('Bos', case=False)
- If df_host returns 0 rows for a valid TAX_ID, host has no occurrence records
- After dataset queries, ALWAYS enrich with wikipedia_search and pubmed_search
  for comprehensive biological context (taxonomy, pathogenesis, epidemiology)

═══════════════════════════════════════════════
TOOL SELECTION RULES
═══════════════════════════════════════════════

1. Geographic / spatial question ("where", "map", "distribution", "location", "detected in")
   → ALWAYS use `create_map`. Never answer with text coordinates.

2. Chart / graph / plot request ("show", "visualize", "pie chart", "bar chart", "histogram")
   → Use `create_visualization`.
   → If the data must be prepared first, call `query_dataframe` before `create_visualization`.

3. Quantitative or tabular question ("how many", "list", "count", "which", "compare")
   → Use `query_dataframe`.

4. Biological / taxonomic background knowledge not in the datasets
   → Use `wikipedia_search` or `pubmed_search`.

5. Combine tools when needed.

═══════════════════════════════════════════════
ACRONYM RESOLUTION & EMPTY RESULT HANDLING
═══════════════════════════════════════════════

- NEVER search with an acronym directly (HIV, HBV, JSRV, SARS, MPOX, etc.)
  ALWAYS resolve the acronym first. Call `wikipedia_search` if unsure.
  Example: HIV → Lentivirus humimdef1 | EBV → Lymphocryptovirus humangamma4

- MANDATORY EMPTY RESULT GUARD:
  After ANY `query_dataframe` or `create_map` call, check if the result is empty (0 rows).
  If empty: try broader term, synonyms, partial name with str.contains().
  Report "no data found" only after at least 2 retry attempts.

═══════════════════════════════════════════════
DATA INTEGRITY RULES
═══════════════════════════════════════════════

- NEVER invent species, families, counts, coordinates, or any biological fact.
- NEVER use column names not listed above. Column names are case-sensitive.
- NEVER display raw column names or dataset structure in your final response.
- Always filter with `.str.contains()` (case-insensitive) rather than `==`.
- ALWAYS include ID in hover_data for map points or plots.
- Report data EXACTLY as returned — no interpretation, no extrapolation.
- NEVER fabricate PMIDs, DOIs, author names, journal names, or publication years. Citations must come exclusively from pubmed_search tool output. A fake PMID is worse than no citation.
- If information is absent from datasets and tools, respond:
  "This information is not available in the current dataset or sources."

═══════════════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════════════

- Scientific, concise, neutral tone.
- Start directly with the answer — no preamble, no "Sure!", no "Great question!".
- No speculation. No storytelling. No unsolicited context.
- Answer ONLY what was asked.
- ANSWER IN MARKDOWN 
- NEVER include image tags, HTML, or Markdown image syntax in your response.
- NEVER invent, guess, or extrapolate a PMID. Only cite a PMID if it was explicitly returned by the pubmed_search tool in this conversation. Hallucinated PMIDs are a critical scientific integrity violation. If no pubmed_search was called, do NOT mention any PMID at all.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]

    used_sources       = set()
    used_wikipedia_urls = []
    used_pubmed_urls    = []
    executed_codes      = []
    generated_figures   = []
    tool_call_count     = 0
    query_cache         = {}
    last_tool_key       = None
    real_pmids          = set()   # PMIDs confirmed by actual pubmed_search calls

    def _sw(icon, label, ok=None):
        if status_container is None:
            return
        prefix = {None: "⏳", True: "✅", False: "❌"}[ok]
        status_container.write(f"{prefix} {icon} {label}{'…' if ok is None else ''}")

    logger.info("=" * 50)
    logger.info(f"USER_QUERY | {user_query}")
    logger.info(
        f"CONFIG | model={model} temp={temperature} top_p={top_p} "
        f"max_calls={max_tool_calls} preview={preview_rows}rows "
        f"max_tool_content={max_tool_content}"
    )

    _sw("🧠", "Thinking")

    while True:
        # ── Call Albert API ──────────────────────────────────────────────────
        logger.info(
            f"LLM_CALL | sending {len(messages)} messages | "
            f"roles={[m['role'] for m in messages]}"
        )
        try:
            resp = _albert_chat(
                messages=messages,
                tools=TOOLS_SPEC,
                model=model,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
            )
        except requests.exceptions.Timeout:
            logger.error("ALBERT_TIMEOUT")
            return (
                "The model took too long to respond (>180 s). Please try again.",
                generated_figures, used_sources,
                used_wikipedia_urls, used_pubmed_urls, executed_codes,
            )
        except Exception as e:
            logger.error(f"ALBERT_ERROR | {e}")
            return (
                f"Could not reach Albert API: {e}",
                generated_figures, used_sources,
                used_wikipedia_urls, used_pubmed_urls, executed_codes,
            )

        # ── Parse OpenAI-format response ─────────────────────────────────────
        choice      = resp["choices"][0]
        msg         = choice["message"]
        finish      = choice.get("finish_reason", "")

        logger.info(f"LLM_RESPONSE | finish_reason={finish!r} | "
                    f"has_tool_calls={bool(msg.get('tool_calls'))} | "
                    f"content_len={len(msg.get('content') or '')}")
        logger.debug(f"LLM_RAW_MSG | {json.dumps(msg, ensure_ascii=False)[:1000]}")

        # ── No tool calls → candidate final answer ───────────────────────────
        if finish != "tool_calls" or not msg.get("tool_calls"):
            final_text = (msg.get("content") or "").strip()

            # ⚠️  gpt-oss-120b bug: early stop with no content and few tool calls
            # — push the model to continue searching before declaring done
            if not final_text and tool_call_count < 3 and tool_call_count > 0:
                logger.warning(
                    f"EARLY_STOP | finish=stop, content empty, only {tool_call_count} tool calls — "
                    f"injecting continuation prompt"
                )
                messages.append(msg)
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have only searched {tool_call_count} time(s) so far and haven't "
                        f"found enough information yet. Please continue your research: "
                        f"search Wikipedia and PubMed for viruses infecting the requested host, "
                        f"then provide a complete scientific answer."
                    ),
                })
                continue  # re-enter the loop

            # ⚠️  gpt-oss-120b bug: returns finish_reason="stop" with empty content
            #    after tool calls. The model cannot synthesize inside a conversation
            #    that contains tool_call messages in its history.
            #    Fix: rebuild a CLEAN conversation with tool results as plain text context.
            if not final_text and tool_call_count > 0:
                logger.warning(
                    "EMPTY_FINAL_ANSWER | content blank after tool calls — "
                    "rebuilding clean context for synthesis (gpt-oss-120b workaround)"
                )

                # Collect all tool results from history (role == "tool")
                tool_results_text = []
                for m in messages:
                    if m.get("role") == "tool":
                        tool_name = m.get("name", "tool")
                        tool_results_text.append(
                            f"=== Result from {tool_name} ===\n{m.get('content', '')}"
                        )

                context_block = "\n\n".join(tool_results_text)
                logger.info(
                    f"SYNTHESIS_CONTEXT | {len(tool_results_text)} tool results | "
                    f"total_chars={len(context_block)}"
                )

                # Evaluate context quality: if most results are errors, say so clearly
                error_count = sum(1 for r in tool_results_text if r.startswith("=== Result") and "Error:" in r)
                success_count = len(tool_results_text) - error_count
                logger.info(f"SYNTHESIS_QUALITY | success={success_count} error={error_count} total={len(tool_results_text)}")

                if success_count == 0 and error_count > 0:
                    # All tool results are errors — context is useless for synthesis
                    # Fall back to pure knowledge answer
                    logger.warning("SYNTHESIS_FALLBACK | all tool results are errors — using model knowledge only")
                    context_note = (
                        f"Note: the dataset did not contain data for this query "
                        f"(search returned no results). Answer from scientific knowledge only."
                    )
                else:
                    context_note = (
                        f"Here is all the information gathered:\n\n{context_block}"
                    )

                # Fresh conversation — NO tool_call messages, NO tools parameter
                clean_messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Original question: {user_query}\n\n"
                            f"{context_note}\n\n"
                            f"Write a detailed, well-structured scientific answer. "
                            f"Requirements:\n"
                            f"- Cover each relevant virus with its taxonomy (family, genus, species), "
                            f"transmission route, pathogenesis, and key clinical signs\n"
                            f"- Use proper scientific nomenclature\n"
                            f"- Structure your answer with one section per virus\n"
                            f"- Be thorough and complete — do not summarize\n"
                            f"- Do not mention tools, datasets, or data retrieval\n"
                            f"- CRITICAL: do NOT invent or mention any PMID. "
                            f"Only reference PMIDs that appear verbatim in the context above. "
                            f"If no PMIDs are in the context, write none."
                        ),
                    },
                ]

                try:
                    retry_resp = _albert_chat(
                        messages=clean_messages,
                        tools=[],        # no tools — force plain text answer
                        model=model,
                        api_key=api_key,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=6144, # synthesis needs more tokens than tool calls
                    )
                    final_text = (
                        retry_resp["choices"][0]["message"].get("content") or ""
                    ).strip()
                    # Strip hallucinated PMIDs from synthesis output
                    final_text, stripped = _strip_hallucinated_pmids(final_text, real_pmids)
                    if stripped:
                        logger.warning(
                            f"PMID_HALLUCINATION_SYNTHESIS | stripped {len(stripped)} fake PMID(s): {stripped}"
                        )
                    logger.info(f"SYNTHESIS_OK | content_len={len(final_text)}")
                    if not final_text:
                        logger.error("SYNTHESIS_FAIL | still empty after clean context")
                        final_text = (
                            "⚠️ The model retrieved information but failed to generate "
                            "a final answer. Please try rephrasing your question."
                        )
                except Exception as e:
                    logger.error(f"SYNTHESIS_RETRY_FAIL | {e}")
                    final_text = (
                        "The model did not produce a final answer. "
                        "Please rephrase your question."
                    )

            # ── PMID hallucination guard ─────────────────────────────────────
            final_text, stripped = _strip_hallucinated_pmids(final_text, real_pmids)
            if stripped:
                logger.warning(
                    f"PMID_HALLUCINATION | stripped {len(stripped)} fake PMID(s): {stripped}"
                )

            logger.info(
                f"RESULT | len={len(final_text)} | "
                f"{final_text[:500]}{'...' if len(final_text) > 500 else ''}"
            )
            messages.append(msg)
            return (
                final_text,
                generated_figures, used_sources,
                used_wikipedia_urls, used_pubmed_urls, executed_codes,
            )

        # Append assistant message (with tool_calls) to conversation history
        messages.append(msg)

        # ── Tool call limit guard ────────────────────────────────────────────
        if tool_call_count >= max_tool_calls:
            logger.warning(f"MAX_TOOL_CALLS | Limit reached ({max_tool_calls})")
            messages.append({
                "role": "system",
                "content": (
                    f"Tool call limit reached ({max_tool_calls}). "
                    f"Synthesize a final answer to: '{user_query}'"
                ),
            })
            continue

        # ── Execute each tool call ───────────────────────────────────────────
        for call in msg["tool_calls"]:
            tool_call_count += 1
            name     = call["function"]["name"]
            raw_args = call["function"].get("arguments", {})
            # ⚠️  Albert/vLLM returns arguments as a JSON *string* — must parse
            args     = _parse_tool_arguments(raw_args)

            logger.info(
                f"TOOL_CALL #{tool_call_count}/{max_tool_calls} | {name} | "
                f"call_id={call.get('id','?')} | "
                f"raw_args_type={type(raw_args).__name__} | "
                f"parsed_args={json.dumps(args, ensure_ascii=False)[:400]}"
            )

            # Redundancy detection — skip and return a memo to the model
            tool_key = (name, json.dumps(args, sort_keys=True))
            if tool_key == last_tool_key:
                logger.warning(
                    f"TOOL_REDUNDANT | {name} identical args as previous call — skipping"
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": call["id"],
                    "name":         name,
                    "content":      "This exact tool call was already executed. Use the previous result.",
                })
                continue
            last_tool_key = tool_key

            icon, label = TOOL_LABELS.get(name, ("🔧", name))
            _sw(icon, label)

            # ── Dispatch ─────────────────────────────────────────────────────
            if name == "wikipedia_search":
                output = wikipedia_search(**args, wikipedia_limit=wikipedia_limit)
                if output["success"]:
                    used_sources.add("Wikipedia")
                    used_wikipedia_urls.append(output["url"])
                    fuzzy_note = (
                        f"\n\n> ⚠️ No exact page found for *{output['original_search']}*"
                        " — showing closest match."
                        if output.get("fuzzy_match") else ""
                    )
                    content = (
                        f"**{output['title']}**\n\n{output['extract']}"
                        f"{fuzzy_note}\n\n🔗 {output['url']}"
                    )
                    logger.info(f"TOOL_OK | wikipedia_search | url={output['url']}")
                    _sw(icon, label, ok=True)
                else:
                    content = output["message"]
                    logger.warning(f"TOOL_FAIL | wikipedia_search | {output['message']}")
                    _sw(icon, label, ok=False)

            elif name == "pubmed_search":
                output = pubmed_search(**args)
                if output["success"]:
                    used_sources.add("PubMed")
                    parts = [f"Found {output['count']} relevant articles on PubMed:\n"]
                    for i, article in enumerate(output["articles"], 1):
                        used_pubmed_urls.append(article["url"])
                        if article.get("pmid"):
                            real_pmids.add(str(article["pmid"]))  # track confirmed PMIDs
                        authors_str = (
                            ", ".join(article["authors"])
                            if article["authors"] else "Unknown authors"
                        )
                        doi_str = f"DOI: {article['doi']}" if article["doi"] else ""
                        parts.append(
                            f"\n--- Article {i} ---\n"
                            f"**{article['title']}**\n"
                            f"Authors: {authors_str} et al.\n"
                            f"Journal: {article['journal']} ({article['year']})\n"
                            f"PMID: {article['pmid']} {doi_str}\n\n"
                            f"Abstract:\n{article['abstract']}\n"
                            f"🔗 {article['url']}"
                        )
                    content = "\n".join(parts)
                    logger.info(f"TOOL_OK | pubmed_search | {output['count']} articles")
                    _sw(icon, label, ok=True)
                else:
                    content = output["message"]
                    logger.warning(f"TOOL_FAIL | pubmed_search | {output['message']}")
                    _sw(icon, label, ok=False)

            elif name == "query_dataframe":
                code      = args.get("code", args.get("_raw", ""))
                cache_key = hash(code)
                if cache_key in query_cache:
                    logger.warning("TOOL_REDUNDANT_QUERY | reusing cached result")
                    output = query_cache[cache_key]
                else:
                    output = query_dataframe(code, df_taxo, df_host, preview_rows=preview_rows)
                    query_cache[cache_key] = output

                if output["success"]:
                    executed_codes.append(code)
                    used_sources.add("Dataset query")
                    content = (
                        f"Query OK. Shape: {output['shape']}\n"
                        f"Columns: {', '.join(output['columns'])}\n"
                        f"{output['preview']}"
                    )
                    logger.info(f"TOOL_OK | query_dataframe | shape={output['shape']}")
                    _sw(icon, label, ok=True)
                else:
                    err_msg = output["message"]
                    # Detect host-name-in-df_host mistake and give targeted hint
                    host_hint = ""
                    if "0 rows" in err_msg and (
                        "df_host" in code and "str.contains" in code
                        and "TAX_ID" not in code
                    ):
                        host_hint = (
                            "\n\nHINT: df_host does not store host species names — "
                            "only numeric TAX_IDs. Use the TWO-STEP approach:\n"
                            "  1. host_taxid = df_taxo[df_taxo['SPECIES_NAME']"
                            ".str.contains('NAME', case=False)]['TAX_ID'].iloc[0]\n"
                            "  2. result = df_host[df_host['<TAX_ID_COL>'] == host_taxid]"
                        )
                        logger.warning("TOOL_HINT | Detected host-name-in-df_host mistake — hint sent to model")
                    content = f"Error:\n{err_msg}{host_hint}"
                    logger.warning(f"TOOL_FAIL | query_dataframe | {err_msg}")
                    _sw(icon, label, ok=False)

            elif name == "create_map":
                code   = args.get("code", args.get("_raw", ""))
                output = create_map(code, df_taxo, df_host)
                if output["success"]:
                    executed_codes.append(code)
                    used_sources.add("Dataset map")
                    generated_figures.append(output["figure"])
                    content = "Map created successfully."
                    if "num_points" in output:
                        content = f"Map created successfully with {output['num_points']} points."
                    logger.info("TOOL_OK | create_map")
                    _sw(icon, label, ok=True)
                else:
                    content = f"Error:\n{output['message']}"
                    logger.warning(f"TOOL_FAIL | create_map | {output['message']}")
                    _sw(icon, label, ok=False)

            elif name == "create_visualization":
                code   = args.get("code", args.get("_raw", ""))
                output = create_visualization(code, df_taxo, df_host)
                if output["success"]:
                    executed_codes.append(code)
                    used_sources.add("Dataset visualization")
                    generated_figures.append(output["figure"])
                    content = "Visualization created successfully."
                    logger.info("TOOL_OK | create_visualization")
                    _sw(icon, label, ok=True)
                else:
                    content = f"Error:\n{output['message']}"
                    logger.warning(f"TOOL_FAIL | create_visualization | {output['message']}")
                    _sw(icon, label, ok=False)

            else:
                content = f"Unknown tool: {name}"
                logger.error(f"UNKNOWN_TOOL | {name}")
                _sw("🔧", name, ok=False)

            # Log tool result preview (before truncation)
            logger.info(
                f"TOOL_RESULT | {name} | total_chars={len(content)} | "
                f"preview={repr(content[:300].replace(chr(10), ' '))}"
            )

            # Truncate oversized tool output before sending back to the model
            if len(content) > max_tool_content:
                original_len = len(content)
                content = (
                    content[:max_tool_content]
                    + f"\n\n[...truncated — {original_len - max_tool_content} chars omitted]"
                )
                logger.warning(
                    f"TOOL_CONTENT_TRUNCATED | {name} | "
                    f"trimmed from {original_len} to {max_tool_content} chars"
                )

            # Append tool result in OpenAI format
            messages.append({
                "role":         "tool",
                "tool_call_id": call["id"],
                "name":         name,
                "content":      content,
            })

        # ── Log conversation state at end of each agentic turn ───────────────
        logger.info(
            f"LOOP_STATE | messages_in_history={len(messages)} | "
            f"tool_calls_used={tool_call_count}/{max_tool_calls} | "
            f"figures={len(generated_figures)} | codes={len(executed_codes)}"
        )


# ==================== MAIN ==================== #

def main():
    st.set_page_config(page_icon="🦠", layout="wide")

    if st.secrets.get("PASSWORD_ENABLED", False):
        check_password()

    # Retrieve API key once (st.stop() if missing)
    api_key = _get_api_key()

    st.title("Welcome :) ")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1400px; }
        .stChatMessage {
            border-radius: 14px; padding: 0.75rem;
            cursor: pointer; transition: background-color 0.2s;
        }
        .stChatMessage:hover { background-color: rgba(100,126,234,0.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.caption("""
        For best results:
        - Use English and provide as much relevant detail as you can
        - Ask precise and clearly formulated questions, avoid acronym or abbreviation
        - Stupid question leads to stupid answer ;)

        Example questions:
        - "Give me information about Orthopoxvirus. Is it a family or a genus? How many species does it include?"
        - "Show a pie chart of genus distribution within Poxviridae."
        - "World repartition of poxviridae"
        - "Tell me more about Polyomavirus infection way"
    """)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("ViromeChat-AI 🦠")
        st.caption("""
A conversational interface to explore viral metagenomic data in the Virome@tlas project.

Ask about:
- Taxonomy information: families, genera, species
- Virus-host relationships and geographic distribution
- Interactive maps and charts
- Additional information via PubMed abstracts

Databases available:
- Taxonomy — NCBI/SRA
- Virus-host occurrences — SRA
- Wikipedia, PubMed (via search tool)
""")

        st.header("Settings")

        # Load datasets
        df_taxo = load_dataframe(TAXO_DB_PATH)
        df_host = load_dataframe(HOST_DB_PATH)
        if df_taxo is None or df_host is None:
            st.error("Required dataset not found")
            st.stop()

        # Fetch available models from Albert
        with st.spinner("Loading models…"):
            model_names = _list_albert_models(api_key)

        with st.expander("⚙️ Expert mode", expanded=False):
            default_idx = next(
                (i for i, m in enumerate(model_names)
                 if ALBERT_MODEL_DEFAULT.lower() in m.lower()),
                0,
            )
            model = st.selectbox(
                "LLM Model", options=model_names, index=default_idx,
                help="Text-generation models available on Albert API",
            )

            st.markdown("**Sampling**")
            temperature = st.slider("Temperature", 0.0, 2.0, DEFAULT_TEMPERATURE, 0.05)
            top_p       = st.slider("Top-p",       0.0, 1.0, DEFAULT_TOP_P,       0.05)

            st.markdown("**Agent**")
            max_tool_calls   = st.slider("Max tool calls",         1,    20,    DEFAULT_MAX_TOOL_CALLS,
                                          help="Increase for complex multi-step questions")
            preview_rows     = st.slider("Preview rows",           5,    200,   DEFAULT_PREVIEW_ROWS, 5)
            wikipedia_limit  = st.slider("Wiki limit (chars/article)", 500, 30000, DEFAULT_WIKIPEDIA_LIMIT, 500)
            max_tool_content = st.slider("Max tool content (chars)",2000, 30000, DEFAULT_MAX_TOOL_CONTENT, 1000)

        st.markdown("---")
        st.caption(f"🔗 GitHub: {GITHUB_URL}")

    # ── Session state ─────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Chat history display ──────────────────────────────────────────────────
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if not msg.get("content"):
                continue
            st.markdown(msg["content"])

            for fig_idx, fig in enumerate(msg.get("figures", [])):
                st.plotly_chart(
                    fig,
                    key=f"fig_{msg_idx}_{fig_idx}",
                    config={"displayModeBar": True, "scrollZoom": True},
                )

            render_sources(
                msg.get("wikipedia_urls", []),
                msg.get("pubmed_urls", []),
                msg.get("executed_codes", []),
            )

            if msg["role"] == "assistant":
                question = ""
                if msg_idx > 0 and st.session_state.messages[msg_idx - 1]["role"] == "user":
                    question = st.session_state.messages[msg_idx - 1].get("content", "")
                render_report_button(
                    msg_idx=msg_idx,
                    question=question,
                    answer=msg["content"],
                    executed_codes=msg.get("executed_codes", []),
                )

    # ── Chat input ────────────────────────────────────────────────────────────
    if query := st.chat_input("Ask about viruses..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_container   = status_placeholder.status("Processing…", expanded=True)

            answer, figures, used_sources, wikipedia_urls, pubmed_urls, executed_codes = (
                albert_agent_loop(
                    model            = model,
                    api_key          = api_key,
                    user_query       = query,
                    df_taxo          = df_taxo,
                    df_host          = df_host,
                    temperature      = temperature,
                    top_p            = top_p,
                    max_tool_calls   = max_tool_calls,
                    preview_rows     = preview_rows,
                    wikipedia_limit  = wikipedia_limit,
                    max_tool_content = max_tool_content,
                    status_container = status_container,
                )
            )

            status_placeholder.empty()
            st.markdown(answer)

            new_msg_idx = len(st.session_state.messages)
            for fig_idx, fig in enumerate(figures):
                st.plotly_chart(
                    fig,
                    key=f"fig_{new_msg_idx}_{fig_idx}",
                    config={"displayModeBar": True, "scrollZoom": True},
                )

            render_sources(wikipedia_urls, pubmed_urls, executed_codes)
            render_report_button(
                msg_idx       = new_msg_idx,
                question      = query,
                answer        = answer,
                executed_codes= executed_codes,
            )

        st.session_state.messages.append({
            "role":           "assistant",
            "content":        answer,
            "figures":        figures,
            "wikipedia_urls": wikipedia_urls,
            "pubmed_urls":    pubmed_urls,
            "executed_codes": executed_codes,
        })

    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** — Results may contain errors and "
        "should be verified for scientific or medical use."
    )


if __name__ == "__main__":
    main()