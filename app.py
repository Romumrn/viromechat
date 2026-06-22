#!/usr/bin/python3
"""
Viromech@t: A conversational interface for exploring viral metagenomic data.
Uses Albert API (OpenAI-compatible) with tool-calling capabilities for:
- Dataset queries (taxonomy and host databases)
- Wikipedia and PubMed searches
- Data visualization and mapping
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

# Local imports
from prompt import build_system_prompt
from tools import (
    wikipedia_search, pubmed_search, TOOL_LABELS,
    query_dataframe, create_visualization, create_map, TOOLS_SPEC,
)
from config import (
    
    ALBERT_BASE_URL, ALBERT_TIMEOUT, ALBERT_MODEL_DEFAULT,
    TAXO_DB_PATH, HOST_DB_PATH, LOG_DIR,
    PAGE_ICON, GITHUB_URL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P,DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PARALLEL_TOOL_CALLS,DEFAULT_SEED,
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CONTENT,
    DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT,
)
from logging_utils import setup_logger


# ── Logging & Error Reporting ─────────────────────────────────────────────────
REPORT_DIR = os.path.join(LOG_DIR, "error_reports")
logger = setup_logger(LOG_DIR)


# ==================== PMID HALLUCINATION GUARD ====================

def _strip_hallucinated_pmids(text: str, real_pmids: set) -> tuple[str, list]:
    """
    Remove PMID references that were hallucinated by the model (not from actual PubMed searches).
    
    Args:
        text: The model's output text to clean
        real_pmids: Set of PMIDs that were actually returned by pubmed_search calls
    
    Returns:
        Tuple of (cleaned_text, list_of_removed_pmids)
    """
    # Match PMID references in various formats: PMID:12345678, PMID #12345678, etc.
    pattern = re.compile(r'\bPMID[:\s#]*([0-9]{5,9})\b', re.IGNORECASE)
    removed = []
    
    def _replace(match):
        pmid = match.group(1)
        if pmid in real_pmids:
            return match.group(0)  # Keep real PMIDs
        removed.append(pmid)
        return ""  # Remove hallucinated PMIDs
    
    cleaned = pattern.sub(_replace, text)
    
    # Clean up orphaned punctuation and references left by PMID removal
    cleaned = re.sub(r'\(e\.g\.\s*,?\s*on\s+[^)]{0,80}\)', '', cleaned)
    cleaned = re.sub(r'\(see\s*\)', '', cleaned)
    
    # Normalize whitespace: collapse multiple spaces/tabs but preserve newlines
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Preserve markdown structure: max 2 consecutive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    cleaned = cleaned.strip()
    return cleaned, removed


# ==================== PASSWORD PROTECTION ====================

def check_password():
    """Simple password gate using Streamlit secrets."""
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


# ==================== DATA LOADING ====================

@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> pd.DataFrame | None:
    """Load CSV data with caching. Returns None if file not found."""
    if not os.path.exists(path):
        logger.error(f"DATA_MISSING | {path}")
        return None
    return pd.read_csv(path)


# ==================== ALBERT API HELPERS ====================

def _get_api_key() -> str:
    """
    Retrieve Albert API key with priority:
    1. st.secrets["ALBERT_API_KEY"]
    2. Environment variable ALBERT_API_KEY
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
    """Build authorization headers for Albert API requests."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _list_albert_models(api_key: str) -> list:
    """
    Fetch available text-generation models from Albert API.
    Filters out embedding, audio, and reranking models.
    Falls back to ALBERT_MODEL_DEFAULT if the API call fails.
    """
    try:
        r = requests.get(
            f"{ALBERT_BASE_URL}/models",
            headers=_albert_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        
        # Filter to text-generation models only
        excluded_keywords = ("embed", "whisper", "rerank")
        names = [
            m["id"] for m in data
            if m.get("object") == "model"
            and not any(kw in m["id"].lower() for kw in excluded_keywords)
        ]
        return names if names else [ALBERT_MODEL_DEFAULT]
        
    except Exception as e:
        logger.warning(f"MODEL_LIST_FAIL | {e} — using default model")
        return [ALBERT_MODEL_DEFAULT]


def _parse_tool_arguments(raw_args) -> dict:
    """
    Parse tool call arguments from Albert API response.
    
    Handles multiple formats that Albert/vLLM may return:
    - dict (ideal case)
    - JSON string (most common)
    - Malformed/partial JSON (known gpt-oss-120b bug)
    
    Falls back to {"_raw": raw_args} if all parsing fails.
    """
    # Case 1: Already a dict
    if isinstance(raw_args, dict):
        return raw_args
    
    # Case 2: Valid JSON string
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass
        
        # Case 3: Malformed JSON — extract key-value pairs with regex
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
        
        # Case 4: Complete failure
        logger.error(f"TOOL_ARG_PARSE_FAIL | raw={raw_args[:300]}")
        return {"_raw": raw_args}
    
    return {}


def _albert_chat(
    messages: list,
    tools: list,
    model: str,
    api_key: str,
    temperature,
    top_p,
    presence_penalty=0,
    frequency_penalty=0,
    seed=42,
    max_completion_tokens=4096,
    parallel_tool_calls=False,
    retry: int = 3,
) -> dict:
    """
    Send chat completion request to Albert API with retry logic.
    
    Retries up to `retry` times on HTTP 429 (rate limiting) with exponential backoff.
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",

        "temperature": temperature,
        "top_p": top_p,

        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,

        "seed": seed,

        "max_completion_tokens": max_completion_tokens,

        "parallel_tool_calls": parallel_tool_calls,

        "stream": False,
    }

    for attempt in range(1, retry + 1):
        try:
            r = requests.post(
                f"{ALBERT_BASE_URL}/chat/completions",
                headers=_albert_headers(api_key),
                json=payload,
                timeout=ALBERT_TIMEOUT,
            )
            
            # Handle rate limiting with exponential backoff
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


# ==================== ERROR REPORTING ====================

def save_error_report(question: str, answer: str, executed_codes: list, comment: str = ""):
    """
    Save an error report with context for debugging.
    
    Includes:
    - User's question and model's answer
    - Executed code snippets
    - User's comment about what went wrong
    - Recent log entries for context
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{ts_str}.json")

    # Collect recent logs for debugging context
    log_filename = os.path.join(LOG_DIR, f"agent_{timestamp.strftime('%Y-%m')}.log")
    related_logs = []
    if os.path.exists(log_filename):
        try:
            with open(log_filename, "r", encoding="utf-8") as f:
                related_logs = [line.rstrip() for line in f.readlines()[-200:]]
        except Exception:
            related_logs = ["[Could not read log file]"]

    report = {
        "timestamp": timestamp.isoformat(),
        "user_comment": comment,
        "question": question,
        "answer": answer,
        "executed_codes": executed_codes,
        "recent_logs": related_logs,
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.warning(f"ERROR_REPORT | Saved to {report_path} | comment={comment!r}")
    return report_path


# ==================== UI COMPONENTS ====================

def render_sources(wikipedia_urls: list, pubmed_urls: list, executed_codes: list):
    """Display expandable sources section with Wikipedia, PubMed, and code references."""
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


def render_report_button(msg_idx: int, question: str, answer: str, executed_codes: list):
    """
    Render an error reporting button with dialog for user feedback.
    
    Tracks whether a report has already been submitted for this message
    to prevent duplicate reports.
    """
    report_key = f"reported_{msg_idx}"
    dialog_key = f"show_dialog_{msg_idx}"

    # Don't show button if already reported
    if st.session_state.get(report_key):
        st.caption("⚠️ Error reported — thank you for your feedback.")
        return

    # Toggle report dialog
    if st.button("🚩 Report an error", key=f"btn_report_{msg_idx}",
                 help="Signal a wrong or misleading answer"):
        st.session_state[dialog_key] = not st.session_state.get(dialog_key, False)

    # Show report dialog
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


# ==================== AGENT LOOP (Core Logic) ====================

def albert_agent_loop(
    model: str,
    api_key: str,
    user_query: str,
    df_taxo: pd.DataFrame,
    df_host: pd.DataFrame,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT,
    max_tool_content: int = DEFAULT_MAX_TOOL_CONTENT,
    status_container=None,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    seed: int = DEFAULT_SEED,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    parallel_tool_calls: bool = DEFAULT_PARALLEL_TOOL_CALLS,
):
    """
    Main agentic loop: iteratively calls Albert API with tool-calling capability.
    
    The loop continues until:
    - The model returns a final answer (no tool calls)
    - Maximum tool call limit is reached
    - An error occurs
    
    Returns:
        Tuple of (answer_text, figures, sources, wikipedia_urls, pubmed_urls, executed_codes)
    """
    # ── Build system prompt with dataset schema ──────────────────────────────
    df_taxo_columns_str = ", ".join(df_taxo.columns)
    df_host_columns_str = ", ".join(df_host.columns)

    def _col_samples(df, col, n=6):
        """Get sample values for a column to help the model understand the data."""
        vals = df[col].dropna().unique()[:n]
        return ", ".join(str(v) for v in vals)

    taxo_samples = {c: _col_samples(df_taxo, c) for c in df_taxo.columns}
    host_samples = {c: _col_samples(df_host, c) for c in df_host.columns}

    taxo_col_detail = "\n".join(f"  - {c}: e.g. {taxo_samples[c]}" for c in df_taxo.columns)
    host_col_detail = "\n".join(f"  - {c}: e.g. {host_samples[c]}" for c in df_host.columns)

    system_prompt = build_system_prompt(taxo_col_detail, host_col_detail)

    # ── Initialize conversation ──────────────────────────────────────────────
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # Track sources and results for display
    used_wikipedia_urls = []
    used_pubmed_urls = []
    executed_codes = []
    generated_figures = []
    real_pmids = set()  # Confirmed PMIDs from actual pubmed_search calls
    
    tool_call_count = 0
    query_cache = {}  # Avoid redundant dataframe queries
    last_tool_key = None  # Detect redundant tool calls

    def _update_status(icon, label, ok=None):
        """Update the status display in the UI."""
        if status_container is None:
            return
        prefix = {None: "⏳", True: "✅", False: "❌"}[ok]
        status_container.write(f"{prefix} {icon} {label}{'…' if ok is None else ''}")

    # ── Logging ──────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info(f"USER_QUERY | {user_query}")
    logger.info(
        f"CONFIG | model={model} temp={temperature} top_p={top_p} "
        f"max_calls={max_tool_calls} preview={preview_rows}rows "
        f"max_tool_content={max_tool_content}"
    )

    _update_status("🧠", "Thinking")

    # ── Main agent loop ──────────────────────────────────────────────────────
    while True:
        logger.info(
            f"LLM_CALL | sending {len(messages)} messages | "
            f"roles={[m['role'] for m in messages]}"
        )
        
        # Call Albert API
        try:
            resp = _albert_chat(
                messages=messages,
                tools=TOOLS_SPEC,
                model=model,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,      # Ajouté
                frequency_penalty=frequency_penalty,    # Ajouté
                seed=seed,                              # Ajouté
                max_completion_tokens=max_completion_tokens,  # Ajouté
                parallel_tool_calls=parallel_tool_calls,      # Ajouté
            )
        except requests.exceptions.Timeout:
            logger.error("ALBERT_TIMEOUT")
            return (
                "The model took too long to respond (>180 s). Please try again.",
                generated_figures, set(), [], [], executed_codes,
            )
        except Exception as e:
            logger.error(f"ALBERT_ERROR | {e}")
            return (
                f"Could not reach Albert API: {e}",
                generated_figures, set(), [], [], executed_codes,
            )

        # Parse response
        choice = resp["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "")

        logger.info(
            f"LLM_RESPONSE | finish_reason={finish!r} | "
            f"has_tool_calls={bool(msg.get('tool_calls'))} | "
            f"content_len={len(msg.get('content') or '')}"
        )

        # ── CASE 1: Final answer (no tool calls) ─────────────────────────────
        if finish != "tool_calls" or not msg.get("tool_calls"):
            final_text = (msg.get("content") or "").strip()

            # Handle gpt-oss-120b bug: early stop with no content
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
                continue

            # Handle empty answer after tool calls (model synthesis workaround)
            if not final_text and tool_call_count > 0:
                logger.warning(
                    "EMPTY_FINAL_ANSWER | content blank after tool calls — "
                    "rebuilding clean context for synthesis (gpt-oss-120b workaround)"
                )

                # Collect all tool results from conversation history
                tool_results_text = []
                for m in messages:
                    if m.get("role") == "tool":
                        tool_name = m.get("name", "tool")
                        tool_results_text.append(
                            f"=== Result from {tool_name} ===\n{m.get('content', '')}"
                        )

                context_block = "\n\n".join(tool_results_text)
                
                # Evaluate context quality
                error_count = sum(
                    1 for r in tool_results_text 
                    if r.startswith("=== Result") and "Error:" in r
                )
                success_count = len(tool_results_text) - error_count

                if success_count == 0 and error_count > 0:
                    # All searches failed — use model knowledge only
                    logger.warning("SYNTHESIS_FALLBACK | all tool results are errors")
                    context_note = (
                        "Note: the dataset did not contain data for this query "
                        "(search returned no results). Answer from scientific knowledge only."
                    )
                else:
                    context_note = f"Here is all the information gathered:\n\n{context_block}"

                # Build fresh conversation for synthesis (no tool messages)
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
                        tools=[],  # Force plain text answer
                        model=model,
                        api_key=api_key,
                        temperature=temperature,
                        top_p=top_p,
                        max_completion_tokens=6144,
                    )
                    final_text = (
                        retry_resp["choices"][0]["message"].get("content") or ""
                    ).strip()
                    
                    # Remove hallucinated PMIDs
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
                    final_text = "The model did not produce a final answer. Please rephrase your question."

            # Final PMID check before returning
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
                generated_figures,
                used_wikipedia_urls,
                used_pubmed_urls,
                executed_codes,
            )

        # ── CASE 2: Tool calls ───────────────────────────────────────────────
        messages.append(msg)

        # Check tool call limit
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

        # Execute each tool call
        for call in msg["tool_calls"]:
            tool_call_count += 1
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments", {})
            args = _parse_tool_arguments(raw_args)

            logger.info(
                f"TOOL_CALL #{tool_call_count}/{max_tool_calls} | {name} | "
                f"call_id={call.get('id','?')} | "
                f"parsed_args={json.dumps(args, ensure_ascii=False)[:400]}"
            )

            # ── Redundancy detection ─────────────────────────────────────────
            tool_key = (name, json.dumps(args, sort_keys=True))
            if tool_key == last_tool_key:
                logger.warning(f"TOOL_REDUNDANT | {name} identical args as previous call — skipping")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": "This exact tool call was already executed. Use the previous result.",
                })
                continue
            last_tool_key = tool_key

            icon, label = TOOL_LABELS.get(name, ("🔧", name))
            _update_status(icon, label)

            # ── Tool Dispatch ────────────────────────────────────────────────
            if name == "wikipedia_search":
                output = wikipedia_search(**args, wikipedia_limit=wikipedia_limit)
                
                if output["success"]:
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
                    _update_status(icon, label, ok=True)
                else:
                    content = output["message"]
                    logger.warning(f"TOOL_FAIL | wikipedia_search | {output['message']}")
                    _update_status(icon, label, ok=False)

            elif name == "pubmed_search":
                output = pubmed_search(**args)
                
                if output["success"]:
                    parts = [f"Found {output['count']} relevant articles on PubMed:\n"]
                    for i, article in enumerate(output["articles"], 1):
                        used_pubmed_urls.append(article["url"])
                        if article.get("pmid"):
                            real_pmids.add(str(article["pmid"]))  # Track for hallucination guard
                        
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
                    _update_status(icon, label, ok=True)
                else:
                    content = output["message"]
                    logger.warning(f"TOOL_FAIL | pubmed_search | {output['message']}")
                    _update_status(icon, label, ok=False)

            elif name == "query_dataframe":
                code = args.get("code", args.get("_raw", ""))
                cache_key = hash(code)
                
                # Use cached result if available
                if cache_key in query_cache:
                    logger.warning("TOOL_REDUNDANT_QUERY | reusing cached result")
                    output = query_cache[cache_key]
                else:
                    output = query_dataframe(code, df_taxo, df_host, preview_rows=preview_rows)
                    query_cache[cache_key] = output

                if output["success"]:
                    executed_codes.append(code)
                    content = (
                        f"Query OK. Shape: {output['shape']}\n"
                        f"Columns: {', '.join(output['columns'])}\n"
                        f"{output['preview']}"
                    )
                    logger.info(f"TOOL_OK | query_dataframe | shape={output['shape']}")
                    _update_status(icon, label, ok=True)
                else:
                    err_msg = output["message"]
                    # Detect common mistake: querying host name in df_host instead of using TAX_ID
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
                        logger.warning("TOOL_HINT | Detected host-name-in-df_host mistake")
                    content = f"Error:\n{err_msg}{host_hint}"
                    logger.warning(f"TOOL_FAIL | query_dataframe | {err_msg}")
                    _update_status(icon, label, ok=False)

            elif name == "create_map":
                code = args.get("code", args.get("_raw", ""))
                output = create_map(code, df_taxo, df_host)
                
                if output["success"]:
                    executed_codes.append(code)
                    generated_figures.append(output["figure"])
                    content = "Map created successfully."
                    if "num_points" in output:
                        content = f"Map created successfully with {output['num_points']} points."
                    logger.info("TOOL_OK | create_map")
                    _update_status(icon, label, ok=True)
                else:
                    content = f"Error:\n{output['message']}"
                    logger.warning(f"TOOL_FAIL | create_map | {output['message']}")
                    _update_status(icon, label, ok=False)

            elif name == "create_visualization":
                code = args.get("code", args.get("_raw", ""))
                output = create_visualization(code, df_taxo, df_host)
                
                if output["success"]:
                    executed_codes.append(code)
                    generated_figures.append(output["figure"])
                    content = "Visualization created successfully."
                    logger.info("TOOL_OK | create_visualization")
                    _update_status(icon, label, ok=True)
                else:
                    content = f"Error:\n{output['message']}"
                    logger.warning(f"TOOL_FAIL | create_visualization | {output['message']}")
                    _update_status(icon, label, ok=False)

            else:
                content = f"Unknown tool: {name}"
                logger.error(f"UNKNOWN_TOOL | {name}")
                _update_status("🔧", name, ok=False)

            # ── Truncate oversized tool output ───────────────────────────────
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

            # Append tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": content,
            })

        # Log loop state at end of each iteration
        logger.info(
            f"LOOP_STATE | messages_in_history={len(messages)} | "
            f"tool_calls_used={tool_call_count}/{max_tool_calls} | "
            f"figures={len(generated_figures)} | codes={len(executed_codes)}"
        )


# ==================== MAIN APPLICATION ====================

def main():
    """Main Streamlit application entry point."""
    st.set_page_config(page_icon="🦠", layout="wide")

    # Password protection (if enabled in secrets)
    if st.secrets.get("PASSWORD_ENABLED", False):
        check_password()

    # Retrieve API key (will stop if not found)
    api_key = _get_api_key()

    # ── Page header ──────────────────────────────────────────────────────────
    st.title("Welcome to Viromech@t")
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

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Viromech@t🦠")
        st.caption("""
A chatbot to explore viral metagenomic data in the Virome@tlas project.

Ask about:
- Taxonomy information: families, genera, species
- Virus-host relationships and geographic distribution
- Interactive maps and charts
- Additional information via PubMed abstractsDatabases available:
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

        # Fetch available models from Albert API
        with st.spinner("Loading models…"):
            model_names = _list_albert_models(api_key)

        # Expert configuration panel
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

            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                DEFAULT_TEMPERATURE,
                0.05
            )

            top_p = st.slider(
                "Top-p",
                0.0, 1.0,
                DEFAULT_TOP_P,
                0.05
            )

            presence_penalty = st.slider(
                "Presence penalty",
                -2.0, 2.0,
                DEFAULT_PRESENCE_PENALTY,
                0.1,
                help="Encourage le modèle à explorer de nouveaux sujets."
            )

            frequency_penalty = st.slider(
                "Frequency penalty",
                -2.0, 2.0,
                DEFAULT_FREQUENCY_PENALTY,
                0.1,
                help="Réduit les répétitions."
            )

            seed = st.number_input(
                "Seed",
                value=DEFAULT_SEED,
                step=1
            )

            max_completion_tokens = st.number_input(
                "Max completion tokens",
                min_value=512,
                max_value=32768,
                value=DEFAULT_MAX_COMPLETION_TOKENS,
                step=512
            )

            parallel_tool_calls = st.checkbox(
                "Parallel tool calls",
                value=DEFAULT_PARALLEL_TOOL_CALLS
            )

            st.markdown("**Agent**")
            max_tool_calls = st.slider(
                "Max tool calls", 1, 20, DEFAULT_MAX_TOOL_CALLS,
                help="Increase for complex multi-step questions"
            )
            preview_rows = st.slider("Preview rows", 5, 200, DEFAULT_PREVIEW_ROWS, 5)
            wikipedia_limit = st.slider(
                "Wiki limit (chars/article)", 500, 30000, DEFAULT_WIKIPEDIA_LIMIT, 500
            )
            max_tool_content = st.slider(
                "Max tool content (chars)", 2000, 30000, DEFAULT_MAX_TOOL_CONTENT, 1000
            )

        st.markdown("---")
        st.caption(f"🔗 GitHub: {GITHUB_URL}")

    # ── Initialize chat history ──────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Display chat history ─────────────────────────────────────────────────
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if not msg.get("content"):
                continue
            st.markdown(msg["content"])

            # Display any generated figures
            for fig_idx, fig in enumerate(msg.get("figures", [])):
                st.plotly_chart(
                    fig,
                    key=f"fig_{msg_idx}_{fig_idx}",
                    config={"displayModeBar": True, "scrollZoom": True},
                )

            # Display sources used
            render_sources(
                msg.get("wikipedia_urls", []),
                msg.get("pubmed_urls", []),
                msg.get("executed_codes", []),
            )

            # Error reporting for assistant messages
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

    # ── Chat input ───────────────────────────────────────────────────────────
    if query := st.chat_input("Ask about viruses..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate assistant response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_container = status_placeholder.status("Processing…", expanded=True)

            answer, figures, wikipedia_urls, pubmed_urls, executed_codes = (
                albert_agent_loop(
                    model=model,
                    api_key=api_key,
                    user_query=query,
                    df_taxo=df_taxo,
                    df_host=df_host,
                    temperature=temperature,
                    top_p=top_p,
                    max_tool_calls=max_tool_calls,
                    preview_rows=preview_rows,
                    wikipedia_limit=wikipedia_limit,
                    max_tool_content=max_tool_content,
                    status_container=status_container,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    seed=seed,
                    max_completion_tokens=max_completion_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                )
            )

            status_placeholder.empty()
            st.markdown(answer)

            # Display generated figures
            new_msg_idx = len(st.session_state.messages)
            for fig_idx, fig in enumerate(figures):
                st.plotly_chart(
                    fig,
                    key=f"fig_{new_msg_idx}_{fig_idx}",
                    config={"displayModeBar": True, "scrollZoom": True},
                )

            # Display sources and error reporting
            render_sources(wikipedia_urls, pubmed_urls, executed_codes)
            render_report_button(
                msg_idx=new_msg_idx,
                question=query,
                answer=answer,
                executed_codes=executed_codes,
            )

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "figures": figures,
            "wikipedia_urls": wikipedia_urls,
            "pubmed_urls": pubmed_urls,
            "executed_codes": executed_codes,
        })

    # ── Footer disclaimer ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** — Results may contain errors and "
        "should be verified for scientific or medical use."
    )


if __name__ == "__main__":
    main()