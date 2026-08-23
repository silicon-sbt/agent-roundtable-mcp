from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roundtable.agent_llm_config import (  # noqa: E402
    DEFAULT_DESCRIPTION,
    load_agent_llm_config_document,
    save_agent_llm_config_document,
)
from roundtable.graph import run_roundtable  # noqa: E402
from llm import (  # noqa: E402
    API_PROVIDERS,
    CLI_BACKENDS,
    CLI_EFFORT_CHOICES,
    MockLLM,
)
from llm_gateway import (  # noqa: E402
    example_model_ids,
    list_provider_model_ids,
    read_model_example,
)
from roundtable.loader import load_council_personas  # noqa: E402

st.set_page_config(
    page_title="Agent Roundtable",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        background: rgba(15, 23, 42, 0.38);
    }
    .agent-row {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 0.85rem 0.9rem 0.35rem 0.9rem;
        margin-bottom: 0.75rem;
        background: rgba(15, 23, 42, 0.24);
    }
    .muted { color: rgba(226, 232, 240, 0.72); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _council_names() -> list[str]:
    names = sorted(path.stem for path in (PROJECT_ROOT / "config" / "councils").glob("*.yaml"))
    return names or ["experts"]


def _select_index(options: list[str], value: str | None) -> int:
    if value in options:
        return options.index(value or "")
    return 0


MODEL_EXAMPLE_PATH = PROJECT_ROOT / "config" / "model_example.json"


@st.cache_data(show_spinner=False)
def _model_example() -> dict[str, Any]:
    return read_model_example(MODEL_EXAMPLE_PATH)


def _route_fragment(route: Any) -> str:
    if not isinstance(route, dict):
        return ""
    provider = str(route.get("provider") or "").strip()
    model = str(route.get("model") or "").strip()
    effort = str(route.get("effort") or "").strip()
    if not provider:
        return ""
    fragment = f"{provider}:{model}" if model else provider
    return f"{fragment}@{effort}" if effort else fragment


@st.cache_data(show_spinner=False)
def _provider_chain_presets() -> list[str]:
    payload = _model_example()
    examples = payload.get("_copy_paste_chains")
    if not isinstance(examples, dict):
        return []
    presets: list[str] = []
    for routes in examples.values():
        if not isinstance(routes, list):
            continue
        fragments = [_route_fragment(route) for route in routes]
        chain = ", ".join(fragment for fragment in fragments if fragment)
        if chain:
            presets.append(chain)
    return list(dict.fromkeys(presets))


PICKER_API_PROVIDERS = [p for p in API_PROVIDERS if p not in ("mock", "auto")]
PICKER_PROVIDERS = PICKER_API_PROVIDERS + list(CLI_BACKENDS)


@st.cache_data(show_spinner=False)
def _curated_models_by_provider() -> dict[str, list[str]]:
    """Offline picker data from the gateway-generated structured example."""

    payload = _model_example()
    return {
        provider: example_model_ids(payload, provider)
        for provider in PICKER_PROVIDERS
        if example_model_ids(payload, provider)
    }


@st.cache_data(show_spinner="Fetching models...", ttl=300)
def _live_models(provider: str) -> tuple[list[str], str, str | None]:
    try:
        models = list_provider_model_ids(provider, project_root=PROJECT_ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        return [], "llm_gateway", f"{type(exc).__name__}: {exc}"
    return models, "llm_gateway", None


def _model_options(provider: str, live: bool) -> tuple[list[str], str]:
    """Models for the dropdown: live fetch (if enabled) merged over hardcoded defaults."""
    curated = _curated_models_by_provider().get(provider, [])
    options = list(curated)
    note = "default (model_example.json)"
    if live:
        models, source, error = _live_models(provider)
        options = list(dict.fromkeys(models + curated))
        note = f"live: {source}" + (f" — {error}" if error else "")
    return options, note


def _append_to_chain(agent_id: str) -> None:
    provider = str(st.session_state.get(f"{agent_id}_pick_provider", "")).strip()
    model = str(st.session_state.get(f"{agent_id}_pick_model", "")).strip()
    effort = str(st.session_state.get(f"{agent_id}_pick_effort", "")).strip()
    if not provider:
        return
    fragment = f"{provider}:{model}" if model and model != "(none)" else provider
    if effort and "@" not in fragment:
        fragment = f"{fragment}@{effort}"
    chain_key = f"{agent_id}_provider_chain"
    current = str(st.session_state.get(chain_key, "")).strip()
    st.session_state[chain_key] = f"{current}, {fragment}" if current else fragment


def _legacy_provider_chain(initial_config: dict[str, Any]) -> str:
    provider_chain = str(initial_config.get("provider_chain") or "").strip()
    if provider_chain:
        return provider_chain
    provider = str(initial_config.get("provider") or "openrouter").strip()
    model = str(initial_config.get("model") or "").strip()
    if provider and model:
        return f"{provider}:{model}"
    return provider


def _render_agent_llm_row(
    agent_id: str,
    display_name: str,
    role: str,
    initial_config: dict[str, Any],
    live_fetch: bool = False,
) -> dict[str, str]:
    st.markdown('<div class="agent-row">', unsafe_allow_html=True)
    heading, chain_col, tuning_col = st.columns([1.25, 3.1, 1.05])
    with heading:
        st.markdown(f"**{display_name}**")
        st.markdown(f'<span class="muted">{agent_id} · {role}</span>', unsafe_allow_html=True)

    current_chain = _legacy_provider_chain(initial_config)
    chain_key = f"{agent_id}_provider_chain"
    if chain_key not in st.session_state:
        st.session_state[chain_key] = current_chain
    current_provider = current_chain.split(",")[0].strip().partition(":")[0]

    with chain_col:
        pick_provider, pick_model, pick_effort, add_col = st.columns([1, 1.7, 0.9, 0.5])
        with pick_provider:
            provider = st.selectbox(
                "Provider",
                PICKER_PROVIDERS,
                index=_select_index(PICKER_PROVIDERS, current_provider),
                key=f"{agent_id}_pick_provider",
            )
        options, note = _model_options(provider, live_fetch)
        display_options = options or ["(none)"]
        model_key = f"{agent_id}_pick_model"
        if st.session_state.get(model_key) not in display_options:
            st.session_state.pop(model_key, None)
        with pick_model:
            st.selectbox("Model", display_options, key=model_key)
        effort_options = CLI_EFFORT_CHOICES.get(provider, ("",))
        effort_key = f"{agent_id}_pick_effort"
        if st.session_state.get(effort_key) not in effort_options:
            st.session_state.pop(effort_key, None)
        with pick_effort:
            st.selectbox("Effort", effort_options, key=effort_key)
        with add_col:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            st.button(
                "➕",
                key=f"{agent_id}_add",
                help="Append provider:model@effort to the chain below",
                on_click=_append_to_chain,
                args=(agent_id,),
                use_container_width=True,
            )
        st.caption(f"Source: {note}")
        provider_chain = st.text_area(
            "Provider chain (editable — comma-separated fallback order)",
            height=72,
            key=chain_key,
        )

    with tuning_col:
        temperature = st.number_input(
            "Temp",
            min_value=0.0,
            max_value=2.0,
            value=float(initial_config.get("temperature", 0.3)),
            step=0.1,
            key=f"{agent_id}_temperature",
        )
        max_output_tokens = st.number_input(
            "Max tokens",
            min_value=256,
            max_value=32768,
            value=int(initial_config.get("max_output_tokens", 4096)),
            step=256,
            key=f"{agent_id}_max_output_tokens",
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "provider_chain": provider_chain,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }


def _merge_agent_configs(
    current_agents: dict[str, dict[str, Any]],
    updates: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    merged = {agent_id: dict(config) for agent_id, config in current_agents.items()}
    for agent_id, config in updates.items():
        merged[agent_id] = {
            key: value
            for key, value in config.items()
            if value is not None and value != ""
        }
    return merged


def _save_configs(agents: dict[str, dict[str, Any]], description: str) -> Path:
    return save_agent_llm_config_document(
        PROJECT_ROOT,
        agents,
        description=description or DEFAULT_DESCRIPTION,
    )


def _progress_event_line(event: dict[str, Any]) -> str:
    event_name = str(event.get("event", "event")).upper()
    stage = str(event.get("stage", ""))
    label = str(event.get("label", ""))
    return f"{event_name} [{stage}] {label}".strip()


st.title("Agent Roundtable")
st.caption("Configure per-agent LLM routing, run a real roundtable, and read the generated Markdown report.")

config_document = load_agent_llm_config_document(PROJECT_ROOT)
saved_agent_configs: dict[str, dict[str, Any]] = config_document["agents"]

with st.sidebar:
    st.header("Run Setup")
    council_names = _council_names()
    council_name = st.selectbox(
        "Council",
        council_names,
        index=_select_index(council_names, "experts"),
    )
    rounds = st.number_input("Rounds", min_value=1, max_value=10, value=1, step=1)
    use_mock = st.toggle("Mock mode", value=False)
    last_result = st.session_state.get("last_result")
    continue_previous = st.toggle(
        "Continue previous transcript",
        value=False,
        disabled=not bool(last_result),
    )
    live_fetch = st.toggle(
        "Fetch live model lists",
        value=False,
        help=(
            "On: query each provider's /models endpoint using .env keys to populate the "
            "Model dropdowns. Off: use the generated config/model_example.json snapshot. "
            "The Provider chain text box is always editable either way."
        ),
    )
    st.caption("Real LLM calls use provider chains in config/agent_llms.json and keys in .env.")

council, personas = load_council_personas(council_name, PROJECT_ROOT)

metric_cols = st.columns(3)
metric_cols[0].metric("Council", council.name)
metric_cols[1].metric("Agents", len(personas))
metric_cols[2].metric("Mode", "Mock" if use_mock else "Real LLM")

st.subheader("Agent LLM Routing")
st.caption("Use provider:model@effort chains. Direct API providers read .env keys; Claude/Codex use official local CLIs; Codex Proxy/Grok/Antigravity use CLIProxyAPI HTTP.")

updates: dict[str, dict[str, str]] = {}
for persona in personas:
    initial = {**persona.llm_config, **saved_agent_configs.get(persona.id, {})}
    updates[persona.id] = _render_agent_llm_row(
        persona.id,
        persona.name,
        persona.role,
        initial,
        live_fetch=live_fetch,
    )

merged_agent_configs = _merge_agent_configs(saved_agent_configs, updates)

save_col, json_col = st.columns([1, 3])
with save_col:
    if st.button("Save LLM JSON", type="primary", use_container_width=True):
        path = _save_configs(merged_agent_configs, str(config_document.get("description", "")))
        st.success(f"Saved: {path.relative_to(PROJECT_ROOT)}")
with json_col:
    with st.expander("Preview config/agent_llms.json"):
        st.json(
            {
                "description": str(config_document.get("description", DEFAULT_DESCRIPTION)),
                "agents": merged_agent_configs,
            },
            expanded=False,
        )
    with st.expander("Model examples from config/model_example.json"):
        st.json(_provider_chain_presets(), expanded=False)

st.divider()
st.subheader("Run Roundtable")

default_topic = "AI 对投资和就业的长期影响"
topic = st.text_area("Topic", value=default_topic, height=90)

run_disabled = not topic.strip()
if st.button("Run With Current Config", type="primary", disabled=run_disabled):
    total_steps = 1 + int(rounds) * (len(personas) + 2) + 2
    progress_state = {"completed": 0}
    progress_events: list[dict[str, Any]] = []
    progress_bar = st.progress(0.0, text="Preparing roundtable run...")
    status_slot = st.empty()
    event_log_slot = st.empty()

    def render_progress_log() -> None:
        if not progress_events:
            return
        event_log_slot.code(
            "\n".join(_progress_event_line(event) for event in progress_events[-12:]),
            language="text",
        )

    def mark_step_done(label: str) -> None:
        progress_state["completed"] = min(progress_state["completed"] + 1, total_steps)
        progress_bar.progress(
            min(progress_state["completed"] / total_steps, 1.0),
            text=label,
        )
        status_slot.success(label)

    def handle_progress(event: dict[str, Any]) -> None:
        progress_events.append(event)
        label = str(event.get("label") or event.get("stage") or "Running")
        if event.get("event") == "done":
            mark_step_done(label)
        else:
            progress_bar.progress(
                min(progress_state["completed"] / total_steps, 0.99),
                text=label,
            )
            status_slot.info(label)
        render_progress_log()

    try:
        saved_path = _save_configs(
            merged_agent_configs,
            str(config_document.get("description", "")),
        )
        save_label = f"Saved current LLM routing to {saved_path.relative_to(PROJECT_ROOT)}"
        progress_events.append(
            {
                "event": "done",
                "stage": "save_config",
                "label": save_label,
            }
        )
        mark_step_done(save_label)
        render_progress_log()

        result = run_roundtable(
            topic=topic,
            council_name=council_name,
            rounds=int(rounds),
            llm=MockLLM() if use_mock else None,
            root_dir=PROJECT_ROOT,
            output_dir=PROJECT_ROOT / "logs",
            progress_callback=handle_progress,
            initial_messages=last_result.get("messages", []) if continue_previous and last_result else None,
            initial_round_summaries=(
                last_result.get("round_summaries", []) if continue_previous and last_result else None
            ),
        )
    except Exception as exc:
        progress_events.append(
            {
                "event": "error",
                "stage": "run_roundtable",
                "label": str(exc),
            }
        )
        progress_bar.progress(
            min(progress_state["completed"] / total_steps, 1.0),
            text="Run failed",
        )
        status_slot.error(f"Run failed: {exc}")
        render_progress_log()
        st.exception(exc)
        st.stop()

    progress_bar.progress(1.0, text="Run complete")
    st.session_state["last_result"] = result
    status_slot.success("Run complete")
    st.success(f"Log saved to {Path(result['log_path']).relative_to(PROJECT_ROOT)}")
    st.markdown("### Final Summary")
    st.markdown(result.get("final_summary", ""))
    with st.expander("Progress Events"):
        st.json(progress_events)
    with st.expander("Transcript Messages"):
        for message in result.get("messages", []):
            speaker = message.get("speaker", "Unknown")
            llm_provider = message.get("llm_provider", "unknown")
            llm_model = message.get("llm_model", "unknown")
            st.markdown(f"**{speaker}** · `{llm_provider}` / `{llm_model}`")
            st.markdown(str(message.get("content", "")))
            st.divider()
