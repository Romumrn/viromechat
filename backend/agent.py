"""
backend/agent.py — the agentic loop, an async generator of structured events.

Rather than returning a single tuple at the end, the loop `yield`s events as
they happen, which the chat endpoint streams to the browser over SSE:

    {"type": "status",      "text": str}
    {"type": "tool_call",   "name": str, "keyword": str, "icon": str, "label": str}
    {"type": "tool_result", "name": str, "ok": bool}
    {"type": "figure",      "figure": <plotly json dict>}
    {"type": "sources",     "wikipedia": [...], "pubmed": [...], "ncbi": [...],
                             "executed_codes": [...]}
    {"type": "final",       "content": str}
    {"type": "error",       "message": str}
    {"type": "done",        "history": [...], "figures": [...], "wikipedia": [...],
                            "pubmed": [...], "ncbi": [...], "executed_codes": [...]}

The terminal "done" event carries everything the caller needs to persist the
turn (via db.add_message) — figures are plotly JSON dicts there. The loop never
touches the database.
"""

# stdlib: used to serialize/deserialize plotly figures and tool-call arguments
import json

# plotly's IO helpers, used to rebuild a Figure object from the JSON an MCP tool returns
import plotly.io as pio
# HTTP client, used here only to catch its Timeout exception type from albert_chat
import requests
# MCP client used to talk to the tool server (list_tools / call_tool)
from fastmcp import Client

# Runtime configuration / default parameter values, centralized in config.py
from config import (
    MCP_SERVER_URL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_SEED, DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PARALLEL_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CONTENT,
    DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT,
)
# Builds the system prompt text, injected with a description of available datasets
from prompt import build_system_prompt
# ALBERT (the LLM backend) client + assorted helpers
from backend.albert import (
    albert_chat, AlbertRateLimitError,
    mcp_tools_to_openai_spec, unwrap_mcp_result, describe_available_datasets,
    parse_tool_arguments, clean_history_messages,
    strip_hallucinated_pmids, strip_fake_citation_markers,
    snippet, ui_search_keyword, TOOL_LABELS, logger,
)


async def check_mcp_connection() -> bool:
    """True if the MCP server answers a list_tools() call."""
    try:
        # Open a short-lived MCP client connection just to probe availability
        async with Client(MCP_SERVER_URL) as client:
            await client.list_tools()
            return True
    except Exception as e:
        # Any failure (connection refused, timeout, etc.) means the server is down
        logger.error(f"MCP_CONNECTION_FAIL | {e}")
        return False


async def run_agent(
    model: str,
    api_key: str,
    user_query: str,
    username: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT,
    max_tool_content: int = DEFAULT_MAX_TOOL_CONTENT,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    seed: int = DEFAULT_SEED,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    parallel_tool_calls: bool = DEFAULT_PARALLEL_TOOL_CALLS,
    history_messages: list | None = None,
):
    """Async generator running the tool-calling loop against ALBERT + MCP,
    yielding events as they happen (see module docstring)."""

    # Accumulators for citation/source links collected across all tool calls in this turn
    used_wikipedia_urls: list = []
    used_pubmed_urls: list = []
    used_ncbi_urls: list = []
    # Any code/SQL snippets executed by tools, kept so they can be shown/persisted later
    executed_codes: list = []
    generated_figures: list = []   # Plotly Figure objects (for db.add_message)
    # PMIDs that genuinely came back from a tool call — used to strip out any the model invents
    real_pmids: set = set()
    # How many tool calls have been made so far this turn, to enforce max_tool_calls
    tool_call_count = 0
    # Whether we've already nudged the model after an empty premature stop (once per turn)
    continuation_nudged = False

    def _done_event(history):
        # Figures serialized to plotly JSON dicts for the caller / SSE.
        return {
            "type": "done",
            "history": history,
            "figures": [json.loads(f.to_json()) for f in generated_figures],
            "wikipedia": used_wikipedia_urls,
            "pubmed": used_pubmed_urls,
            "ncbi": used_ncbi_urls,
            "executed_codes": executed_codes,
        }

    # --- Log the start of this turn: separator, the raw query, and the run config ---
    logger.info("=" * 50)
    logger.info(f"USER_QUERY | user={username or 'unknown'} | {user_query}")
    logger.info(
        f"CONFIG | model={model} temp={temperature} top_p={top_p} "
        f"max_calls={max_tool_calls} preview={preview_rows}rows "
        f"max_tool_content={max_tool_content}"
    )

    # Let the caller show a "Thinking" indicator before the first LLM round-trip
    yield {"type": "status", "text": "Thinking"}

    # Open the MCP connection for the whole turn (tools are listed once, called many times)
    async with Client(MCP_SERVER_URL) as mcp:
        # Discover the tools this MCP server exposes and convert them to OpenAI's
        # function-calling schema so ALBERT can decide when/how to call them
        tools = await mcp.list_tools()
        tools_spec = mcp_tools_to_openai_spec(tools)
        # Keep the raw input schema per tool name, used later to sanitize call arguments
        tool_schemas = {t.name: (t.inputSchema or {}) for t in tools}

        # Ask the MCP server what datasets it has, then bake that into the system prompt
        # so the model knows what it can query
        datasets_description = await describe_available_datasets(mcp)
        system_prompt = build_system_prompt(datasets_description)

        # Seed the conversation: system prompt, then any prior turns, then the new question
        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": user_query})

        # Main agent loop: keep calling the LLM until it produces a final answer
        # (no more tool calls requested) or an unrecoverable error occurs
        while True:
            logger.info(
                f"LLM_CALL | sending {len(messages)} messages | "
                f"roles={[m['role'] for m in messages]}"
            )

            try:
                # Ask ALBERT for the next step: either a final message or tool_calls
                resp = albert_chat(
                    messages=messages, tools=tools_spec, model=model, api_key=api_key,
                    temperature=temperature, top_p=top_p,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                    seed=seed, max_completion_tokens=max_completion_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                )
            except requests.exceptions.Timeout:
                # ALBERT took too long to respond — surface an error event and stop the turn
                logger.error("ALBERT_TIMEOUT")
                yield {"type": "error",
                       "message": "The model took too long to respond (>120 s). Please try again."}
                yield _done_event(None)
                return
            except AlbertRateLimitError:
                # Free-tier ALBERT API is saturated — tell the user to retry or switch models
                logger.error("ALBERT_RATE_LIMITED")
                yield {"type": "error",
                       "message": ("⏳ Albert is rate-limiting requests right now — the free API is "
                                   "saturated, especially on the large models. Please wait a moment "
                                   "and try again, or pick a lighter model in Expert mode.")}
                yield _done_event(None)
                return
            except Exception as e:
                # Catch-all for network/API errors we didn't anticipate
                logger.error(f"ALBERT_ERROR | {e}")
                yield {"type": "error", "message": f"Could not reach Albert API: {e}"}
                yield _done_event(None)
                return

            # Pull out the single choice ALBERT returned (no multi-completion support here)
            choice = resp["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason", "")

            logger.info(
                f"LLM_RESPONSE | finish_reason={finish!r} | "
                f"has_tool_calls={bool(msg.get('tool_calls'))} | "
                f"content_len={len(msg.get('content') or '')}"
            )

            # ── CASE 1: Final answer (no tool calls) ─────────────────────────
            if finish != "tool_calls" or not msg.get("tool_calls"):
                final_text = (msg.get("content") or "").strip()

                # Guard against the model stopping too early: gpt-oss often stops with
                # empty content mid-investigation (e.g. right after a lookup confirmed a
                # virus name exists, before querying its rows). While tool budget remains,
                # nudge it once to keep going instead of falling back to the synthesis
                # below, which only sees partial results and may wrongly conclude
                # "no data". Once per turn, so an always-empty model can't loop forever.
                if (not final_text and tool_call_count > 0
                        and tool_call_count < max_tool_calls and not continuation_nudged):
                    continuation_nudged = True
                    logger.warning(
                        f"EARLY_STOP | finish=stop, content empty after {tool_call_count} tool calls — "
                        f"injecting continuation prompt"
                    )
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your reply was empty and the original question is not answered yet. "
                            "Continue working on it: call the tools you still need — in particular, "
                            "if a lookup query just confirmed that a name exists in the dataset, now "
                            "run the real query filtering on that exact value (and create the map or "
                            "chart if one was asked for) — then write the complete final answer."
                        ),
                    })
                    # Skip straight to the next loop iteration (another LLM call)
                    continue

                # Some models (notably gpt-oss-120b) sometimes return an empty final
                # message even after doing real tool work. Work around that by
                # re-asking for a synthesis from a clean context built purely from
                # the tool results already gathered, instead of the full message history.
                if not final_text and tool_call_count > 0:
                    logger.warning(
                        "EMPTY_FINAL_ANSWER | content blank after tool calls — "
                        "rebuilding clean context for synthesis (gpt-oss-120b workaround)"
                    )
                    yield {"type": "status", "text": "Synthesizing answer"}

                    # Collect every tool result message into one readable text block
                    tool_results_text = []
                    for m in messages:
                        if m.get("role") == "tool":
                            tool_name = m.get("name", "tool")
                            tool_results_text.append(
                                f"=== Result from {tool_name} ===\n{m.get('content', '')}"
                            )
                    context_block = "\n\n".join(tool_results_text)

                    # Count how many of those results were errors vs. successes, so we can
                    # tell the model whether it actually has usable data or not
                    error_count = sum(
                        1 for r in tool_results_text
                        if r.startswith("=== Result") and "Error:" in r
                    )
                    success_count = len(tool_results_text) - error_count

                    if success_count == 0 and error_count > 0:
                        # Every tool call failed — tell the model to fall back on its own knowledge
                        logger.warning("SYNTHESIS_FALLBACK | all tool results are errors")
                        context_note = (
                            "Note: every query to the virome@tlas database failed or returned no "
                            "results for this question. State that explicitly at the start of the "
                            "answer (the database query did not return data — not that the data "
                            "does not exist), then answer from general scientific knowledge, "
                            "clearly labeled as such."
                        )
                    else:
                        context_note = f"Here is all the information gathered:\n\n{context_block}"

                    # A fresh, minimal message list: just the system prompt plus one
                    # instruction message with the gathered context and strict formatting
                    # / anti-hallucination requirements. No tools are offered this time.
                    clean_messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Original question: {user_query}\n\n"
                                f"{context_note}\n\n"
                                f"Write a detailed, well-structured scientific answer. "
                                f"Requirements:\n"
                                f"- Start with what the virome@tlas database results above show "
                                f"(counts, hosts, countries, dates), attributed to the virome@tlas "
                                f"database; only then add external context (Wikipedia, PubMed) in a "
                                f"separate section\n"
                                f"- Cover each relevant virus with its taxonomy (family, genus, species), "
                                f"transmission route, pathogenesis, and key clinical signs\n"
                                f"- Use proper scientific nomenclature\n"
                                f"- Structure your answer with one section per virus\n"
                                f"- Be thorough and complete — do not summarize\n"
                                f"- Do not mention tool names, raw column names or data retrieval steps\n"
                                f"- CRITICAL: do NOT invent or mention any PMID. "
                                f"Only reference PMIDs that appear verbatim in the context above. "
                                f"If no PMIDs are in the context, write none.\n"
                                f"- CRITICAL: never claim the data contains no records for something "
                                f"if any result above shows it exists (e.g. a lookup returning its "
                                f"name). If the results are incomplete, say the retrieval did not "
                                f"finish and suggest asking again — do not present that as absence of data."
                            ),
                        },
                    ]

                    try:
                        # One-shot synthesis call — no tools, larger token budget for a full write-up
                        retry_resp = albert_chat(
                            messages=clean_messages, tools=[], model=model, api_key=api_key,
                            temperature=temperature, top_p=top_p, max_completion_tokens=6144,
                        )
                        final_text = (
                            retry_resp["choices"][0]["message"].get("content") or ""
                        ).strip()

                        # Even in the synthesis path, strip any PMIDs/citation markers the
                        # model may have hallucinated that don't match real_pmids
                        final_text, stripped = strip_hallucinated_pmids(final_text, real_pmids)
                        if stripped:
                            logger.warning(f"PMID_HALLUCINATION_SYNTHESIS | stripped {len(stripped)}: {stripped}")
                        final_text, n_fake = strip_fake_citation_markers(final_text)
                        if n_fake:
                            logger.warning(f"FAKE_CITATION_SYNTHESIS | stripped {n_fake} marker(s)")

                        logger.info(f"SYNTHESIS_OK | content_len={len(final_text)}")
                        if not final_text:
                            # Synthesis retry also came back empty — give up gracefully
                            logger.error("SYNTHESIS_FAIL | still empty after clean context")
                            final_text = (
                                "⚠️ The model retrieved information but failed to generate "
                                "a final answer. Please try rephrasing your question."
                            )
                    except Exception as e:
                        # Synthesis call itself failed (network/API error) — final fallback message
                        logger.error(f"SYNTHESIS_RETRY_FAIL | {e}")
                        final_text = "The model did not produce a final answer. Please rephrase your question."

                # Whether we got here via the normal path or the synthesis workaround,
                # always run the anti-hallucination cleanup one more time before returning.
                final_text, stripped = strip_hallucinated_pmids(final_text, real_pmids)
                if stripped:
                    logger.warning(f"PMID_HALLUCINATION | stripped {len(stripped)}: {stripped}")
                final_text, n_fake = strip_fake_citation_markers(final_text)
                if n_fake:
                    logger.warning(f"FAKE_CITATION | stripped {n_fake} marker(s)")

                logger.info(
                    f"RESULT | len={len(final_text)} | "
                    f"{final_text[:500]}{'...' if len(final_text) > 500 else ''}"
                )
                # Record the assistant's final message in the running history
                messages.append(msg)

                # Emit a "sources" event if we collected any citations OR ran any
                # code/SQL — the executed snippets belong in the Sources panel too.
                if (used_wikipedia_urls or used_pubmed_urls or used_ncbi_urls
                        or executed_codes):
                    yield {"type": "sources",
                           "wikipedia": used_wikipedia_urls,
                           "pubmed": used_pubmed_urls,
                           "ncbi": used_ncbi_urls,
                           "executed_codes": executed_codes}
                # Emit the answer text, then the terminal "done" event with everything
                # the caller needs to persist (history, figures, sources, executed code)
                yield {"type": "final", "content": final_text}
                yield _done_event(clean_history_messages(messages[1:]))
                return

            # ── CASE 2: Tool calls ───────────────────────────────────────────
            # The model wants to call one or more tools before answering.
            # Record its tool-call request message first.
            messages.append(msg)

            # If we've already hit the tool-call budget, don't execute more tools —
            # instead force the model to answer with whatever it already has.
            if tool_call_count >= max_tool_calls:
                logger.warning(f"MAX_TOOL_CALLS | Limit reached ({max_tool_calls})")
                messages.append({
                    "role": "system",
                    "content": (
                        f"Tool call limit reached ({max_tool_calls}). "
                        f"Synthesize a final answer to: '{user_query}'"
                    ),
                })
                # Loop back around; next LLM call has no tools left to use in practice
                continue

            # Execute every tool call the model requested in this turn, in order
            for call in msg["tool_calls"]:
                tool_call_count += 1
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments", {})
                # Arguments come back from ALBERT as a JSON string (or malformed variants);
                # parse_tool_arguments normalizes them into a plain dict
                args = parse_tool_arguments(raw_args)

                logger.info(
                    f"TOOL_CALL #{tool_call_count}/{max_tool_calls} | {name} | "
                    f"call_id={call.get('id','?')} | "
                    f"parsed_args={json.dumps(args, ensure_ascii=False)[:400]}"
                )

                # Look up a friendly icon/label for this tool (falls back to a generic wrench)
                icon, label = TOOL_LABELS.get(name, ("🔧", name))
                # Best-effort extraction of a human-readable keyword for the UI (e.g. what was searched)
                keyword = ui_search_keyword(args)
                yield {"type": "tool_call", "name": name,
                       "icon": icon, "label": label, "keyword": keyword}

                # Look up this tool's declared input schema so we can sanitize the call
                schema = tool_schemas.get(name) or {}
                props = schema.get("properties", {})
                required = schema.get("required", [])

                call_args = dict(args)
                # Some models emit a single unnamed "_raw" argument instead of the proper
                # named parameter; if the tool has exactly one required param, remap it.
                if set(call_args.keys()) == {"_raw"} and len(required) == 1:
                    call_args[required[0]] = call_args.pop("_raw")

                # Fill in server-side defaults for common optional parameters if the
                # tool supports them and the model didn't specify a value
                if "preview_rows" in props:
                    call_args.setdefault("preview_rows", preview_rows)
                if "wikipedia_limit" in props:
                    call_args.setdefault("wikipedia_limit", wikipedia_limit)

                # Drop any arguments the model invented that aren't in the tool's schema,
                # to avoid the MCP call failing on unexpected kwargs
                if props:
                    dropped = set(call_args) - set(props)
                    if dropped:
                        logger.warning(f"TOOL_ARG_DROPPED | {name} | unsupported keys: {sorted(dropped)}")
                        call_args = {k: v for k, v in call_args.items() if k in props}

                # Actually invoke the MCP tool and unwrap its result into a plain dict
                # with "success"/"content"/"artifacts" keys
                output = unwrap_mcp_result(await mcp.call_tool(name, call_args))
                content = output.get("content", "Unknown error")

                if output.get("success"):
                    # Walk any artifacts the tool returned (links, PMIDs, figures, ...)
                    # and fold them into the turn-level accumulators / emit events for them
                    for artifact in output.get("artifacts", []):
                        a_type = artifact.get("type")
                        if a_type == "url":
                            # Generic web link (e.g. Wikipedia) — dedupe before storing
                            if artifact["url"] not in used_wikipedia_urls:
                                used_wikipedia_urls.append(artifact["url"])
                        elif a_type == "pubmed":
                            # One artifact can carry multiple PMIDs; build canonical PubMed URLs
                            # and also track the raw PMIDs as "real" for hallucination stripping
                            for pmid in artifact.get("pmids", []):
                                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                if pubmed_url not in used_pubmed_urls:
                                    used_pubmed_urls.append(pubmed_url)
                                real_pmids.add(str(pmid))
                        elif a_type == "ncbi_taxonomy":
                            if artifact["url"] not in used_ncbi_urls:
                                used_ncbi_urls.append(artifact["url"])
                        elif a_type == "plotly":
                            # Rehydrate the plotly figure from JSON, keep the Figure object
                            # for later persistence, and stream the raw JSON to the client now
                            fig = pio.from_json(json.dumps(artifact["figure"]))
                            generated_figures.append(fig)
                            yield {"type": "figure", "figure": artifact["figure"]}

                    # Track any code/SQL the tool executed, for later display/audit
                    if "code" in call_args:
                        executed_codes.append(call_args["code"])
                    elif "sql" in call_args:
                        executed_codes.append(f"-- SQL ({name})\n{call_args['sql']}")

                    logger.info(f"TOOL_OK | {name} | {snippet(content, 500)}")
                else:
                    logger.warning(f"TOOL_FAIL | {name} | {content}")

                # Let the UI know this tool call finished, and whether it succeeded
                yield {"type": "tool_result", "name": name, "ok": bool(output.get("success"))}

                # Cap how much tool output we feed back into the conversation, to avoid
                # blowing the context window on huge results (e.g. large query outputs)
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

                # Feed the (possibly truncated) tool result back into the conversation
                # as a "tool" role message, tied to its originating call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": content,
                })

            # End-of-round summary log, then loop back to ask the LLM what to do next
            logger.info(
                f"LOOP_STATE | messages_in_history={len(messages)} | "
                f"tool_calls_used={tool_call_count}/{max_tool_calls} | "
                f"figures={len(generated_figures)} | codes={len(executed_codes)}"
            )
