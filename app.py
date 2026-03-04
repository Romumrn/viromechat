#!/usr/bin/python3
import streamlit as st
import pandas as pd
import requests
import os
import logging
from datetime import datetime 
from tools import wikipedia_search, pubmed_search, TOOL_LABELS, query_dataframe, create_visualization, create_map, TOOLS_SPEC
from config import (
    TAXO_DB_PATH, HOST_DB_PATH, LOG_DIR,
    OLLAMA_BASE_URL, OLLAMA_TIMEOUT, OLLAMA_DEFAULT_MODEL_PREFIX,
    PAGE_ICON, GITHUB_URL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPEAT_PENALTY, DEFAULT_SEED,
    DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CONTENT,
    DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT
)


# ==================== LOGGER SETUP ==================== #

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = f"{LOG_DIR}/agent_{datetime.now().strftime('%Y-%m')}.log"

    logger = logging.getLogger("virus_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


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

# ==================== SOURCES DISPLAY ==================== #

def render_sources(wikipedia_urls, pubmed_urls, executed_codes):
    """Factorised sources expander — used both in chat history and new responses."""
    if not wikipedia_urls and not pubmed_urls and not executed_codes:
        return
    with st.expander("📚 Sources"):
        if wikipedia_urls:
            st.markdown("**📘 Wikipedia**")
            for url in wikipedia_urls:
                title = url.split('/')[-1].replace('_', ' ')
                st.markdown(f"- [{title}]({url})")
        if pubmed_urls:
            st.markdown("**🔬 PubMed**")
            for url in pubmed_urls:
                # Extraire le PMID de l'URL pour l'affichage
                pmid = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
                st.markdown(f"- [PMID: {pmid}]({url})")
        if executed_codes:
            st.markdown("**📊 Dataset Query & Visualization**")
            full_code = "\n\n---\n\n".join(
                f"# Code {i}\n{code}"
                for i, code in enumerate(executed_codes, 1)
            )
            st.code(full_code, language="python")


# ==================== AGENT LOOP ==================== #

def ollama_agent_loop(model, user_query, df_taxo, df_host,
                      temperature=DEFAULT_TEMPERATURE,
                      top_p=DEFAULT_TOP_P,
                      repeat_penalty=DEFAULT_REPEAT_PENALTY,
                      seed=DEFAULT_SEED,
                      max_tool_calls=DEFAULT_MAX_TOOL_CALLS,
                      preview_rows=DEFAULT_PREVIEW_ROWS,
                      wikipedia_limit=DEFAULT_WIKIPEDIA_LIMIT,
                      status_container=None):
    """
    status_container: a st.status() context object. If provided, live tool
    steps are written into it and it is marked complete when done.
    """
    df_taxo_columns_str = ", ".join(df_taxo.columns)
    df_host_columns_str = ", ".join(df_host.columns)

    messages = [
        {
            "role": "system",
            "content": f"""
You are a scientific bioinformatics assistant specialized in virology and viral ecology.
You have access to two curated datasets and external tools. You must ground every statement in data or tool output.

═══════════════════════════════════════════════
AVAILABLE DATA
═══════════════════════════════════════════════

`df_taxo` — Taxonomy database (NCBI/SRA)
  Columns: {df_taxo_columns_str}

`df_host` — Virus-host occurrence database (SRA)
  Columns: {df_host_columns_str}

You can link both dataframe with "TAX_ID" 

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

6. Combine tools when needed: e.g. query_dataframe → create_visualization, or
   wikipedia_search + pubmed_search + query_dataframe for mixed factual + data questions.

═══════════════════════════════════════════════
ACRONYM RESOLUTION & EMPTY RESULT HANDLING
═══════════════════════════════════════════════

- NEVER search with an acronym directly (HIV, HBV, JSRV, SARS, MPOX, etc.)
  ALWAYS resolve the acronym and search with their species name of TAXID 
  If unsure of the full name, call `wikipedia_search` first to resolve it.
  Example: HIV -> Lentivirus humimdef1
  EBV -> Lymphocryptovirus humangamma4

- MANDATORY EMPTY RESULT GUARD:
  After ANY `query_dataframe` or `create_map` call, check if the result is empty (0 rows).
  If the result is empty:
    1. Try a broader search term (e.g. genus instead of species, partial name)
    2. Try alternate spellings or synonyms
    3. Call `query_dataframe` with: result = df_host[df_host['VIRAL_SPECIES'].str.contains('TERM', case=False, na=False)]
       to inspect what names are actually present before retrying the map or query.
    4. Only report "no data found" after at least 2 retry attempts with different terms.
    
═══════════════════════════════════════════════
DATA INTEGRITY RULES
═══════════════════════════════════════════════

- NEVER assume that common virus names exist in the dataset.
- ALWAYS verify species existence in df_taxo before querying df_host.
- NEVER invent species, families, counts, coordinates, or any biological fact.
- NEVER use column names not listed above. Column names are case-sensitive.
- NEVER display raw column names or dataset structure in your final response.
- When using `query_dataframe`, ALWAYS explicitly select the MINIMUM number of columns required to answer the question.
- If a piece of information is absent from the datasets and cannot be retrieved by a tool, respond:
  "This information is not available in the current dataset or sources."
- Report data EXACTLY as returned by the dataset or tool — no interpretation, no extrapolation.
- Always filter with `.str.contains()` (case-insensitive) rather than `==` for species/genus/family name matching.
- ALWAYS include ID in hover_data for map points or plot.

═══════════════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════════════

- Scientific, concise, neutral tone.
- Start directly with the answer — no preamble, no "Sure!", no "Great question!".
- No speculation. No storytelling. No unsolicited context.
- Answer ONLY what was asked. Do not expand to related topics.
- Do not discuss drugs, treatments, cancer, or any non-virology medical topic.
- NEVER include image tags, HTML, Markdown image syntax (![...](...)), or any reference to figure attachments in your response.
- When citing PubMed articles, include the PMID and a brief summary of key findings.
"""
        },
        {"role": "user", "content": user_query}
    ]

    used_sources, used_wikipedia_urls, used_pubmed_urls, executed_codes, generated_figures = set(), [], [], [], []
    tool_call_count = 0

    def _sw(icon, label, ok=None):
        """Write a step line into the st.status container if available.
        ok=None → running (no checkmark), ok=True → ✅, ok=False → ❌
        """
        if status_container is None:
            return
        if ok is None:
            status_container.write(f"⏳ {icon} {label}…")
        elif ok:
            status_container.write(f"✅ {icon} {label}")
        else:
            status_container.write(f"❌ {icon} {label}")

    logger.info("=" * 50)
    logger.info(f"USER_QUERY | {user_query}")
    logger.info(
        f"CONFIG | temp={temperature} top_p={top_p} penalty={repeat_penalty} seed={seed} "
        f"max_calls={max_tool_calls} preview={preview_rows}rows"
    )

    _sw("🧠", "Thinking")

    while True: 
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model, "messages": messages, "tools": TOOLS_SPEC, "stream": False,
                    "options": {
                        "temperature": temperature, "top_p": top_p,
                        "repeat_penalty": repeat_penalty, "seed": seed
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )
            r.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("OLLAMA_TIMEOUT | Model did not respond within 120s")
            return (
                "The model took too long to respond (>120s). Please try again or select a faster model.",
                generated_figures, used_sources, used_wikipedia_urls, used_pubmed_urls, executed_codes
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"OLLAMA_ERROR | {e}")
            return (
                f"Could not reach Ollama: {e}",
                generated_figures, used_sources, used_wikipedia_urls, used_pubmed_urls, executed_codes
            )

        msg = r.json()["message"]

        if thinking := msg.get("thinking"):
            logger.info(f"THINKING | {thinking[:600]}{'...' if len(thinking) > 600 else ''}")

        # ── Stop condition — no more tool calls ──
        if "tool_calls" not in msg:
            messages.append(msg)
            logger.info(f"RESULT | {msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}")
            return msg["content"], generated_figures, used_sources, used_wikipedia_urls, used_pubmed_urls, executed_codes

        messages.append({"role": "assistant", "content": "", "tool_calls": msg["tool_calls"]})

        if tool_call_count >= max_tool_calls:
            logger.warning(f"MAX_TOOL_CALLS | Limit reached ({max_tool_calls}), forcing final answer")
            messages.append({
                "role": "system",
                "content": f"Tool call limit reached ({max_tool_calls}). Synthesize a final answer to: '{user_query}'"
            })
        
        else:

            for call in msg["tool_calls"]:
                tool_call_count += 1
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                logger.info(f"TOOL_CALL #{tool_call_count} | {name} | args={args}")

                icon, label = TOOL_LABELS.get(name, ("🔧", name))
                _sw(icon, label)

                # ── Execute the tool ──
                if name == "wikipedia_search":
                    output = wikipedia_search(**args, wikipedia_limit=wikipedia_limit)
                    if output["success"]:
                        used_sources.add("Wikipedia")
                        used_wikipedia_urls.append(output["url"])
                        fuzzy_note = ""
                        if output.get("fuzzy_match"):
                            fuzzy_note = f"\n\n> ⚠️ No exact page found for *{output['original_search']}* — showing closest match."
                        content = f"**{output['title']}**\n\n{output['extract']}{fuzzy_note}\n\n🔗 {output['url']}"
                        logger.info(f"TOOL_OK | wikipedia_search | {output['url']}")
                        _sw(icon, label, ok=True)
                    else:
                        content = output["message"]
                        logger.warning(f"TOOL_FAIL | wikipedia_search | {output['message']}")
                        _sw(icon, label, ok=False)

                elif name == "pubmed_search":
                    output = pubmed_search(**args)
                    if output["success"]:
                        used_sources.add("PubMed")
                        articles = output["articles"]
                        content_parts = [f"Found {output['count']} relevant articles on PubMed:\n"]
                        
                        for i, article in enumerate(articles, 1):
                            used_pubmed_urls.append(article["url"])
                            authors_str = ", ".join(article["authors"]) if article["authors"] else "Unknown authors"
                            doi_str = f"DOI: {article['doi']}" if article["doi"] else ""
                            
                            content_parts.append(
                                f"\n--- Article {i} ---\n"
                                f"**{article['title']}**\n"
                                f"Authors: {authors_str} et al.\n"
                                f"Journal: {article['journal']} ({article['year']})\n"
                                f"PMID: {article['pmid']} {doi_str}\n\n"
                                f"Abstract:\n{article['abstract']}\n"
                                f"🔗 {article['url']}"
                            )
                        
                        content = "\n".join(content_parts)
                        logger.info(f"TOOL_OK | pubmed_search | found {output['count']} articles")
                        _sw(icon, label, ok=True)
                    else:
                        content = output["message"]
                        logger.warning(f"TOOL_FAIL | pubmed_search | {output['message']}")
                        _sw(icon, label, ok=False)

                elif name == "query_dataframe":
                    output = query_dataframe(args["code"], df_taxo, df_host, preview_rows=preview_rows)
                    if output["success"]:
                        executed_codes.append(args["code"])
                        used_sources.add("Dataset query")
                        content = f"Query OK. Shape: {output['shape']}\nColumns: {', '.join(output['columns'])}\n{output['preview']}"
                        logger.info(f"TOOL_OK | query_dataframe | shape={output['shape']}")
                        _sw(icon, label, ok=True)
                    else:
                        content = f"Error:\n{output['message']}"
                        logger.warning(f"TOOL_FAIL | query_dataframe | {output['message']}")
                        _sw(icon, label, ok=False)

                elif name == "create_map":
                    output = create_map(args["code"], df_taxo, df_host)
                    if output["success"]:
                        executed_codes.append(args["code"])
                        used_sources.add("Dataset map")
                        generated_figures.append(output["figure"])
                        content = "Map created successfully."
                        logger.info("TOOL_OK | create_map")
                        _sw(icon, label, ok=True)
                    else:
                        content = f"Error:\n{output['message']}"
                        logger.warning(f"TOOL_FAIL | create_map | {output['message']}")
                        _sw(icon, label, ok=False)

                elif name == "create_visualization":
                    output = create_visualization(args["code"], df_taxo, df_host)
                    if output["success"]:
                        executed_codes.append(args["code"])
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

                # Truncate tool content to avoid Ollama 500s on large payloads
                MAX_TOOL_CONTENT = DEFAULT_MAX_TOOL_CONTENT
                if len(content) > MAX_TOOL_CONTENT:
                    content = content[:MAX_TOOL_CONTENT] + f"\n\n[...truncated — {len(content) - MAX_TOOL_CONTENT} chars omitted]"
                    logger.warning(f"TOOL_CONTENT_TRUNCATED | {name} | content trimmed to {MAX_TOOL_CONTENT} chars")

                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}
                )


# ==================== MAIN ==================== #

def main():
    st.set_page_config(page_icon="🦠", layout="wide")

    if st.secrets.get("PASSWORD_ENABLED", False):
        check_password()

    st.title("Welcome PRABI :) ")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        .stChatMessage {
            border-radius: 14px;
            padding: 0.75rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .stChatMessage:hover {
            background-color: rgba(100, 126, 234, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.caption("""
        For best results:
        - Use English and provide as much relevant detail as you can
        - Ask precise and clearly formulated questions
        - Double-check that your question is scientifically accurate
        - Stupid question leads to stupid answer :) 

        Example questions:   
        - "Give me information about Orthopoxvirus. Is it a family or a genus? How many species does it include?"
        - "Show a pie chart of genus distribution within Poxviridae."
        - "World repartition of poxvirdae"
        - "Tell me more about Polyomavirus infection way" 
    """)

    with st.sidebar:
        st.title("Virus Dataset AI Agent 🦠 ")
        st.caption("""
Explore the viral world through AI-assisted analysis of curated bioinformatics datasets.

Ask about:
- Taxonomy information: families, genera, species
- Virus-host relationships and geographic distribution
- Interactive maps and charts 
- Additional information via PubMed abstracts

Databases available:
- Taxonomy — NCBI/SRA
- Virus-host occurrences - SRA
- Wikipedia,  PubMed (via search tool)

More coming soon!
""")

        st.header("Settings")

        try:
            ollama_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_TIMEOUT)
            if ollama_response.status_code != 200:
                st.error("Ollama not running")
                st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Make sure Ollama is running.")
            st.stop()

        df_taxo = load_dataframe(TAXO_DB_PATH)
        df_host = load_dataframe(HOST_DB_PATH)

        if df_taxo is None or df_host is None:
            st.error("Required dataset not found")
            st.stop()

        model_names = [m["name"] for m in ollama_response.json().get("models", [])]
        if not model_names:
            st.error("No models available in Ollama")
            st.stop()

        with st.expander("⚙️ Expert mode", expanded=False):
            default_index = next((i for i, m in enumerate(
                model_names) if m.startswith(OLLAMA_DEFAULT_MODEL_PREFIX)), 0)
            model = st.selectbox("LLM Model", options=model_names, index=default_index,
                                 help="Select a model able to use tools")

            st.markdown("**Sampling**")
            temperature = st.slider("Temperature",    0.0, 2.0,   DEFAULT_TEMPERATURE,    0.05)
            top_p       = st.slider("Top-p",          0.0, 1.0,   DEFAULT_TOP_P,          0.05)
            repeat_penalty = st.slider("Repeat penalty", 0.5, 2.0, DEFAULT_REPEAT_PENALTY, 0.05)
            seed = st.number_input("Seed", min_value=-1, max_value=99999, value=DEFAULT_SEED, step=1)
            st.markdown("**Agent**")
            max_tool_calls  = st.slider("Max tool calls",  1,   20,    DEFAULT_MAX_TOOL_CALLS)
            preview_rows    = st.slider("Preview rows",    5,   200,   DEFAULT_PREVIEW_ROWS,   5)
            wikipedia_limit = st.slider("Wiki limit",      500, 20000, DEFAULT_WIKIPEDIA_LIMIT, 500)

        st.markdown("---")
        st.caption(f"🔗 GitHub: {GITHUB_URL}")

    # ── Session state ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Chat history display ──
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            # Skip messages still being generated (no content yet)
            if not msg.get("content"):
                continue
            st.markdown(msg["content"])

            # ── FIX: Stable figure keys using only msg/fig index ──
            for fig_idx, fig in enumerate(msg.get("figures", [])):
                st.plotly_chart(
                    fig,
                    key=f"fig_{msg_idx}_{fig_idx}",
                    config={'displayModeBar': True, 'scrollZoom': True}
                )

            # ── FIX: Factorised sources rendering ──
            render_sources(
                msg.get("wikipedia_urls", []), 
                msg.get("pubmed_urls", []), 
                msg.get("executed_codes", [])
            )

    # ── Chat input ──
    if query := st.chat_input("Ask about viruses..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # ── Spinner visible only during generation, removed after ──
            status_placeholder = st.empty()
            status_container = status_placeholder.status("Processing...", expanded=True)

            answer, figures, used_sources, wikipedia_urls, pubmed_urls, executed_codes = ollama_agent_loop(
                model=model,
                user_query=query,
                df_taxo=df_taxo,
                df_host=df_host,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                max_tool_calls=max_tool_calls,
                preview_rows=preview_rows,
                wikipedia_limit=wikipedia_limit,
                status_container=status_container,
            )

            # Clear the spinner entirely once done
            status_placeholder.empty()

            st.markdown(answer)

            new_msg_idx = len(st.session_state.messages)  # index of the soon-to-be-appended message
            for fig_idx, fig in enumerate(figures):
                st.plotly_chart(
                    fig,
                    key=f"fig_{new_msg_idx}_{fig_idx}",
                    config={'displayModeBar': True, 'scrollZoom': True}
                )
                
            render_sources(wikipedia_urls, pubmed_urls, executed_codes)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "figures": figures,
            "wikipedia_urls": wikipedia_urls,
            "pubmed_urls": pubmed_urls,
            "executed_codes": executed_codes,
        })

    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** — Results may contain errors and should be verified for scientific or medical use.")


if __name__ == "__main__":
    main()