"""Streamlit side of the dashboard agent: the chat sidebar and its result panel.

This module only moves data between three things that do not know about
each other: the agent loop (app/agent/loop.py), the pure UI-state layer
(app/agent/ui_state.py), and Streamlit's session state. The agent's tool
results become a DashboardState; the dashboard then renders that state --
the date picker, the highlighted game, the table/chart panel -- the same
way it would if a person had set those controls.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st
from openai import OpenAIError

from app.agent.loop import client_from_env, configuration_error, run_agent
from app.agent.ui_state import DashboardState, apply_tool_result

STATE_KEY: Final[str] = "agent_state"
CHAT_KEY: Final[str] = "agent_chat"
SLATE_DATE_KEY: Final[str] = "slate_date"


def agent_state() -> DashboardState:
    """Return what the agent last pointed the dashboard at."""

    return st.session_state.setdefault(STATE_KEY, DashboardState())


def run_turn(project_root: Path, chat: list[dict[str, object]], prompt: str) -> None:
    """Run the agent on one request, then apply what it did to session state."""

    client, model = client_from_env()
    history = [{"role": turn["role"], "content": turn["content"]} for turn in chat]

    try:
        with st.spinner("Working on it..."):
            result = run_agent(
                client, model, prompt, project_root, dt.date.today(), history=history
            )
    except OpenAIError as error:
        st.error(f"The model request failed: {error}")
        return

    # slate_date is cleared first so the picker only moves when a tool in
    # this turn returned a date, not because an earlier turn once did.
    state = replace(agent_state(), slate_date=None)

    for call in result.tool_calls:
        state = apply_tool_result(state, call.name, call.result)

    st.session_state[STATE_KEY] = state

    if state.slate_date is not None:
        st.session_state[SLATE_DATE_KEY] = dt.date.fromisoformat(state.slate_date)

    chat.append({"role": "user", "content": prompt})
    chat.append({"role": "assistant", "content": result.answer, "actions": result.tool_calls})


def render_agent_chat(project_root: Path) -> None:
    """Render the chat sidebar and, on a new request, run the agent.

    Must run before the slate date picker is created: a tool result that
    carries a date is applied by writing the picker's session-state key,
    which Streamlit only allows before the widget exists in a run. The
    rest of the page then renders from the updated state in this same
    run -- no st.rerun(), which would drop the picker's own state.
    """

    chat = st.session_state.setdefault(CHAT_KEY, [])

    with st.sidebar:
        st.subheader("🤖 Ask the dashboard")
        st.caption(
            "Ask in plain English. The agent can only use this dashboard's own tools, and "
            "its numbers are model estimates -- not betting advice."
        )
        enabled = configuration_error() is None

        if not enabled:
            st.info(
                "Chat is disabled: no language model is configured for this app. "
                "Set OPENAI_API_KEY and NBA_AGENT_MODEL to enable it (see the README)."
            )

        transcript = st.container()
        prompt = st.chat_input("e.g. Pick the winners for tomorrow's games", disabled=not enabled)

    with transcript:
        if prompt:
            run_turn(project_root, chat, prompt)

        for turn in chat:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

                if turn.get("actions"):
                    with st.expander(f"Actions taken ({len(turn['actions'])})"):
                        for action in turn["actions"]:
                            st.json(
                                {
                                    "tool": action.name,
                                    "arguments": action.arguments,
                                    "result": action.result,
                                },
                                expanded=1,
                            )


def render_agent_view() -> None:
    """Show the table (and chart) for whatever the agent last looked up."""

    panel = agent_state().panel

    if panel is None:
        return

    with st.container(border=True):
        st.markdown(f"**🤖 Agent view: {panel.title}**")

        if not panel.rows:
            st.caption("Nothing to show -- the tool returned no rows.")
            return

        table = pd.DataFrame(panel.rows)
        st.dataframe(table, hide_index=True, width="stretch")

        if panel.chart is not None:
            x_column, y_column = panel.chart
            st.bar_chart(table, x=x_column, y=y_column)
