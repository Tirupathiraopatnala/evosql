"""
app.py — EvoSQL-Lite v3.1

Changes from v3:
  - Fitness displayed as seconds (e.g. "140.5s") everywhere — no abstract numbers
  - Baseline panel shows plan shape (from post-execution EXPLAIN)
  - Generation headers: st.markdown (not expanders) — enables agent cards as expanders
  - Agent cards: st.expander collapsed by default — click to see SQL + plan + strategy
  - Live rewrite panel: agent rewrite-ready shown as collapsed expander (SQL inside)
  - Execution panel: each result shown as collapsed expander with plan metrics + SQL
  - Stage handlers updated: agent_rewrite_ready replaces agent_estimated
  - Analytics chart Y axis = "Execution Time (seconds)"
"""

import streamlit as st
import pandas as pd
import os
import difflib
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

from synapse_client import SynapseClient
from schema_extractor import SchemaExtractor
from evolution import EvolutionEngine
from debug_logger import reset_log

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="EvoSQL-Lite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 EvoSQL-Lite: Evolutionary Query Optimizer")
st.markdown("*Watch AI agents evolve SQL optimizations in real-time*")

load_dotenv()

# ---------------------------
# Config
# ---------------------------
SYNAPSE_SERVER   = "syn-dlr-eda-dev.sql.azuresynapse.net"
SYNAPSE_DB       = "Syndw"
SYNAPSE_USER     = "syn_rw_bi"
SYNAPSE_PASSWORD = os.getenv("SYNAPSE_PASSWORD")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL    = os.getenv("AZURE_OPENAI_MODEL")

# ---------------------------
# Helper: SQL diff view
# ---------------------------
def show_sql_diff(original: str, rewritten: str):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original SQL**")
        st.code(original, language="sql", line_numbers=True)
    with col2:
        st.markdown("**Rewritten SQL**")
        st.code(rewritten or "(none)", language="sql", line_numbers=True)
    with st.expander("📝 View Detailed Diff"):
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            (rewritten or "").splitlines(keepends=True),
            lineterm=""
        )
        st.code("".join(diff), language="diff")


# ---------------------------
# Helper: genome panel
# ---------------------------
def genome_panel(genome_values: dict, genome_instructions: list):
    TRAIT_LABELS = {
        "predicate_pushdown_bias":         "🔽 Predicate Pushdown",
        "shuffle_avoidance_bias":          "🔀 Shuffle Avoidance",
        "broadcast_tolerance":             "📡 Broadcast Tolerance",
        "join_reorder_weight":             "🔄 Join Reorder",
        "temp_table_materialization_bias": "🗄️ Temp Table",
        "aggregation_pushdown_bias":       "➕ Aggregation Push",
        "index_exploitation_bias":         "📇 Index Exploit",
        "partition_elimination_bias":      "✂️ Partition Elim",
    }
    cols = st.columns(4)
    for i, (key, label) in enumerate(TRAIT_LABELS.items()):
        val   = genome_values.get(key, 0.0)
        badge = "🟢" if val > 0.7 else ("🟡" if val > 0.4 else "🔴")
        cols[i % 4].metric(label, f"{badge} {val:.2f}")
    st.markdown("**📋 Strategy instructions sent to LLM:**")
    if genome_instructions:
        for inst in genome_instructions:
            st.markdown(f"- {inst}")
    else:
        st.warning("⚠️ All genome values < 0.4 — using fallback: general best practices")


# ---------------------------
# Helper: plan shape metrics row
# ---------------------------
def render_plan_metrics(plan_metrics: dict):
    if not plan_metrics:
        st.caption("Plan metrics unavailable")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Plan Cost",     f"{plan_metrics.get('total_cost', 0):.0f}")
    c2.metric("🔀 Shuffle Ops",   plan_metrics.get("shuffle_ops",   0))
    c3.metric("📡 Broadcast Ops", plan_metrics.get("broadcast_ops", 0))
    c4.metric("🔍 Full Scans",    plan_metrics.get("full_scans",    0))
    c5.metric("⚠️ Stale Stats",   plan_metrics.get("stale_stats",   0))


# ---------------------------
# Helper: generation feedback panel
# ---------------------------
def render_feedback_panel(generation_context: dict):
    if not generation_context:
        return
    st.markdown("**🧠 Generation Feedback — injected into next generation's LLM prompts:**")
    winner = generation_context.get("winner", {})
    if winner:
        impr = winner.get("improvement", 0)
        sign = "+" if impr >= 0 else ""
        st.markdown(
            f"Best agent: `{winner.get('strategy', '?')}` — "
            f"**{sign}{impr:.2f}% improvement** "
            f"({winner.get('execution_time', '?')}s)"
        )
        diff_summary = winner.get("diff_summary", "")
        if diff_summary:
            st.markdown("**What it changed:**")
            st.code(diff_summary, language="diff")
    failed = generation_context.get("failed_strategies", [])
    if failed:
        st.markdown("**❌ Approaches that didn't help (next gen will avoid):**")
        for f in failed:
            st.markdown(f"  - {f}")
    st.caption("📡 This context is injected into every agent's prompt in the next generation.")


# ---------------------------
# Helper: render completed generation
# (NOT an expander — so agent cards inside CAN be expanders)
# ---------------------------
def render_gen_table(gen_data: dict):
    gen      = gen_data["generation"]
    best_fit = gen_data["best_fitness"]   # seconds
    impr     = gen_data["improvement"]
    pop      = gen_data["population"]
    agent_cards        = gen_data.get("agent_cards", [])
    generation_context = gen_data.get("generation_context")

    sign = "+" if impr >= 0 else ""
    color = "🟢" if impr > 0 else ("🔴" if impr < 0 else "⚪")

    st.markdown(
        f"#### {color} Generation {gen} &nbsp;|&nbsp; "
        f"Best: **{best_fit:.2f}s** &nbsp;|&nbsp; "
        f"Improvement: **{sign}{impr:.2f}%**"
    )

    # Summary table
    rows = []
    for agent in pop:
        fitness = agent["fitness"]
        executed = agent["has_actual_fitness"]
        schema_ok = not agent["schema_removed"]

        if executed:
            time_label = f"✅ {fitness:.2f}s"
        elif fitness >= 9999.0:
            time_label = "❌ Penalized"
        else:
            time_label = "⚠️ Not executed"

        rows.append({
            "Agent":     agent["agent_id"],
            "Strategy":  agent["strategy"],
            "Time":      time_label,
            "Status":    agent["status"],
            "Schema OK": "✅" if schema_ok else "⚠️",
            "Failure":   agent.get("failure_reason") or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Generation feedback
    if generation_context:
        st.markdown("---")
        render_feedback_panel(generation_context)

    # Agent detail cards — each is a collapsed expander
    if agent_cards:
        st.markdown("---")
        st.markdown("**🔬 Agent Details** *(click to expand)*")
        for card in agent_cards:
            sql           = card.get("rewritten_sql")
            strat         = card.get("optimization_strategy", [])
            plan_metrics  = card.get("plan_metrics", {})
            agent_id      = card.get("agent_id", "")
            strategy_name = card.get("strategy_name", "")
            exec_time     = card.get("execution_time")

            if card.get("ok"):
                time_str = f" — ⏱️ {exec_time:.2f}s" if exec_time is not None else ""
                label = f"✅ Agent `{agent_id}` ({strategy_name}){time_str}"
            elif card.get("warn"):
                label = f"⚠️ Agent `{agent_id}` ({strategy_name}) — {card.get('text','')}"
            else:
                label = f"❌ Agent `{agent_id}` ({strategy_name}) — {card.get('text','')}"

            with st.expander(label, expanded=False):
                if plan_metrics:
                    render_plan_metrics(plan_metrics)
                if strat:
                    st.markdown("**Strategy applied:**")
                    for s in strat:
                        st.markdown(f"- {s}")
                if sql:
                    st.code(sql, language="sql", line_numbers=True)

    st.divider()


def rebuild_history():
    if not st.session_state.completed_gens:
        return
    with history_placeholder.container():
        st.markdown("---")
        st.markdown("### 📚 Completed Generations")
        for g in st.session_state.completed_gens:
            render_gen_table(g)


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    pop_size        = st.slider("Population Size",  3, 10, 5)
    num_generations = st.slider("Max Generations",  2, 10, 3)
    st.markdown("---")
    st.markdown("### Status")
    status_placeholder = st.empty()
    status_placeholder.info("Ready to start")

# ---------------------------
# SQL Input
# ---------------------------
sql_input = st.text_area(
    "📝 Paste your SQL query here:",
    height=200,
    placeholder="SELECT ... FROM ... WHERE ..."
)

start_button = st.button(
    "🚀 Start Evolution" if not st.session_state.get("is_running") else "⏳ Evolution Running...",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.get("is_running", False)
)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Live Evolution", "🏆 Results", "📈 Analytics"])

with tab1:
    overview_container = st.container()
    baseline_container = st.container()

with tab2:
    live_placeholder    = st.empty()
    history_placeholder = st.empty()

with tab3:
    winner_container = st.container()

with tab4:
    analytics_container = st.container()

# ---------------------------
# Session State Init
# ---------------------------
if "evolution_log"      not in st.session_state: st.session_state.evolution_log      = []
if "fitness_history"    not in st.session_state: st.session_state.fitness_history    = []
if "failure_patterns"   not in st.session_state: st.session_state.failure_patterns   = {}
if "is_running"         not in st.session_state: st.session_state.is_running         = False
if "pending_sql"        not in st.session_state: st.session_state.pending_sql        = ""
if "schema_cache"       not in st.session_state: st.session_state.schema_cache       = None
if "completed_gens"     not in st.session_state: st.session_state.completed_gens     = []
if "cur_agent_cards"    not in st.session_state: st.session_state.cur_agent_cards    = []
if "execution_cards"    not in st.session_state: st.session_state.execution_cards    = []
if "cur_agent_ids"      not in st.session_state: st.session_state.cur_agent_ids      = []
if "cur_agent_id"       not in st.session_state: st.session_state.cur_agent_id       = None
if "exec_agent_ids"     not in st.session_state: st.session_state.exec_agent_ids     = []
if "exec_agent_id"      not in st.session_state: st.session_state.exec_agent_id      = None
if "generation_context" not in st.session_state: st.session_state.generation_context = None
if "baseline_time"      not in st.session_state: st.session_state.baseline_time      = None
if "exec_start_time"    not in st.session_state: st.session_state.exec_start_time    = None


# ---------------------------
# Live rewrite panel
# ---------------------------
def render_live_gen(gen, total_gens, agent_idx, total_agents,
                    strategy, genome_values, genome_instructions,
                    agent_cards, all_agent_ids=None, current_agent_id=None):
    with live_placeholder.container():
        st.markdown(f"### 🧬 Generation {gen} of {total_gens} — Rewriting Agents")
        progress = agent_idx / total_agents if total_agents else 0
        st.progress(progress, text=f"Agent {agent_idx}/{total_agents}: **{strategy}** calling LLM...")

        # Agent status badges
        if all_agent_ids:
            completed_ids = {c.get("agent_id") for c in agent_cards}
            cols = st.columns(min(len(all_agent_ids), 6))
            for i, aid in enumerate(all_agent_ids):
                short = aid[:8]
                col   = cols[i % len(cols)]
                if aid == current_agent_id:
                    col.markdown(f"⏳ `{short}`\n*rewriting...*")
                elif aid in completed_ids:
                    card = next((c for c in agent_cards if c.get("agent_id") == aid), None)
                    if card and card.get("ok"):
                        col.markdown(f"✅ `{short}`")
                    elif card and card.get("warn"):
                        col.markdown(f"⚠️ `{short}`")
                    else:
                        col.markdown(f"❌ `{short}`")
                else:
                    col.markdown(f"🔘 `{short}`\n*pending*")

        with st.expander("🔬 Genome & LLM Instructions for current agent", expanded=False):
            genome_panel(genome_values, genome_instructions)

        # Completed agents — collapsed expanders
        if agent_cards:
            st.markdown("---")
            st.markdown("**Rewrites ready:**")
            for card in agent_cards:
                sql           = card.get("rewritten_sql")
                strat         = card.get("optimization_strategy", [])
                agent_id      = card.get("agent_id", "")
                strategy_name = card.get("strategy_name", "")

                if card.get("ok"):
                    label = f"✅ `{agent_id}` ({strategy_name}) — ready for execution"
                elif card.get("warn"):
                    label = f"⚠️ `{agent_id}` ({strategy_name}) — {card.get('text','')}"
                else:
                    label = f"❌ `{agent_id}` ({strategy_name}) — {card.get('text','')}"

                with st.expander(label, expanded=False):
                    if strat:
                        st.markdown("**Strategy:**")
                        for s in strat:
                            st.markdown(f"- {s}")
                    if sql:
                        st.code(sql, language="sql")


# ---------------------------
# Live execution panel
# ---------------------------
def render_execution_panel(gen, total_gens, exec_agent_ids, exec_agent_id, execution_cards):
    # Fully clear before rebuilding — prevents rewrite-phase genome expander bleeding through
    live_placeholder.empty()
    with live_placeholder.container():
        st.markdown(f"### 🧬 Generation {gen} of {total_gens} — Executing All Agents")
        st.markdown("### ⚡ Synapse Execution Queue")

        exec_done_ids = {c.get("agent_id") for c in execution_cards}
        if exec_agent_ids:
            cols2 = st.columns(min(len(exec_agent_ids), 6))
            for i, aid in enumerate(exec_agent_ids):
                short = aid[:8]
                col   = cols2[i % len(cols2)]
                if aid == exec_agent_id:
                    elapsed = int(time.time() - st.session_state.get("exec_start_time", time.time()))
                    col.markdown(f"⏳ `{short}`\n*running... {elapsed}s*")
                elif aid in exec_done_ids:
                    card = next((c for c in execution_cards if c.get("agent_id") == aid), None)
                    col.markdown(f"✅ `{short}`" if (card and card.get("ok")) else f"❌ `{short}`")
                else:
                    col.markdown(f"🔘 `{short}`\n*queued*")

        # Execution result cards — collapsed expanders
        if execution_cards:
            st.markdown("**Results so far:**")
            for ec in execution_cards:
                plan_metrics  = ec.get("plan_metrics", {})
                sql           = ec.get("rewritten_sql")
                strat         = ec.get("optimization_strategy", [])
                exec_time     = ec.get("execution_time")
                strategy_name = ec.get("strategy", "")
                agent_id      = ec.get("agent_id", "")

                if ec["ok"]:
                    time_str = f" — ⏱️ {exec_time:.2f}s" if exec_time is not None else ""
                    label = f"✅ `{agent_id[:8]}` ({strategy_name}){time_str}"
                else:
                    label = f"❌ `{agent_id[:8]}` ({strategy_name}) — {ec.get('text','failed')}"

                with st.expander(label, expanded=False):
                    if plan_metrics:
                        render_plan_metrics(plan_metrics)
                    if strat:
                        st.markdown("**Strategy applied:**")
                        for s in strat:
                            st.markdown(f"- {s}")
                    if sql:
                        st.code(sql, language="sql")


# ---------------------------
# Button handler
# ---------------------------
if start_button and sql_input.strip():
    reset_log()
    st.session_state.is_running       = True
    st.session_state.pending_sql      = sql_input
    st.session_state.evolution_log    = []
    st.session_state.fitness_history  = []
    st.session_state.failure_patterns = {}
    st.session_state.completed_gens   = []
    st.session_state.cur_agent_cards  = []
    st.session_state.execution_cards  = []
    st.session_state.cur_agent_ids    = []
    st.session_state.cur_agent_id     = None
    st.session_state.exec_agent_ids   = []
    st.session_state.exec_agent_id    = None
    st.session_state.schema_cache     = None
    st.session_state.generation_context = None
    st.session_state.baseline_time      = None
    st.session_state.exec_start_time    = None
    st.rerun()

# ---------------------------
# Main Evolution Loop
# ---------------------------
if st.session_state.get("is_running") and st.session_state.get("pending_sql"):

    sql_to_run = st.session_state.pending_sql

    try:
        status_placeholder.info("🔌 Connecting to Synapse...")
        synapse = SynapseClient(
            server=SYNAPSE_SERVER, database=SYNAPSE_DB,
            username=SYNAPSE_USER, password=SYNAPSE_PASSWORD
        )
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        if st.session_state.schema_cache is None:
            status_placeholder.info("📋 Loading schema metadata (~10s first run)...")
            schema_extractor = SchemaExtractor(synapse)
            st.session_state.schema_cache = schema_extractor.build_prompt_metadata(sql=sql_to_run)
        schema_metadata = st.session_state.schema_cache

        engine = EvolutionEngine(
            synapse_client=synapse, openai_client=openai_client,
            model_name=AZURE_OPENAI_MODEL,
            population_size=pop_size, generations=num_generations
        )
        status_placeholder.info("🧬 Starting evolution...")

    except Exception as e:
        status_placeholder.error(f"❌ Initialization failed: {e}")
        st.session_state.is_running = False
        st.stop()

    for update in engine.run(sql_to_run, schema_metadata):
        stage = update["stage"]
        st.session_state.evolution_log.append(update)

        # ── Fatal ─────────────────────────────────────────────────
        if stage == "fatal_error":
            status_placeholder.error(f"❌ {update['reason']}")
            st.session_state.is_running = False
            with overview_container:
                st.error("**Evolution could not start**")
                st.markdown(f"**Reason:** `{update['reason']}`")
                st.warning(update["message"])
            break

        # ── Baseline ──────────────────────────────────────────────
        elif stage == "baseline_started":
            status_placeholder.info("⏳ Running baseline query on Synapse...")
            with live_placeholder.container():
                st.info("⏳ Executing baseline query — please wait...")

        elif stage == "baseline_complete":
            metrics      = update["baseline_metrics"]
            baseline_t   = update["baseline_fitness"]   # seconds
            baseline_plan = update.get("baseline_plan", {})
            st.session_state.baseline_time = baseline_t
            status_placeholder.success("✅ Baseline complete")
            st.session_state.fitness_history = [{
                "generation": 0, "label": "Baseline",
                "best_fitness": baseline_t, "improvement": 0.0
            }]
            with baseline_container:
                st.subheader("📊 Baseline Performance")
                c1, c2 = st.columns(2)
                c1.metric("⏱️ Execution Time", f"{baseline_t:.2f}s")
                c2.metric("🔢 Row Count",       metrics.get("row_count", 0))
                if baseline_plan:
                    st.markdown("**📐 Query Plan Shape:**")
                    render_plan_metrics(baseline_plan)

        # ── Generation started ────────────────────────────────────
        elif stage == "generation_started":
            gen = update["generation"]
            st.session_state.cur_agent_cards = []
            st.session_state.cur_agent_ids   = []
            st.session_state.cur_agent_id    = None
            st.session_state.exec_agent_ids  = []
            st.session_state.exec_agent_id   = None
            st.session_state.execution_cards = []
            status_placeholder.info(f"🔄 Gen {gen}/{num_generations} — LLM rewriting agents...")
            with live_placeholder.container():
                st.markdown(f"### 🧬 Generation {gen} of {num_generations}")
                ctx = update.get("generation_context")
                if ctx:
                    st.markdown("---")
                    render_feedback_panel(ctx)
                    st.markdown("---")
                st.progress(0.0, text="Preparing agents...")

        # ── Agent rewriting ───────────────────────────────────────
        elif stage == "agent_rewriting":
            gen           = update["generation"]
            agent_idx     = update["agent_index"]
            total         = update["total_agents"]
            strategy      = update["strategy"]
            agent_id      = update.get("agent_id", "")
            genome_values = update.get("genome_values", {})
            genome_instrs = update.get("genome_instructions", [])

            if agent_id and agent_id not in st.session_state.cur_agent_ids:
                st.session_state.cur_agent_ids.append(agent_id)
            st.session_state.cur_agent_id = agent_id

            status_placeholder.info(
                f"🔄 Gen {gen}/{num_generations} — "
                f"Agent {agent_idx}/{total}: {strategy} calling LLM..."
            )
            render_live_gen(gen, num_generations, agent_idx, total,
                            strategy, genome_values, genome_instrs,
                            st.session_state.cur_agent_cards,
                            all_agent_ids=st.session_state.cur_agent_ids,
                            current_agent_id=agent_id)

        # ── Agent rewrite ready (LLM + safety passed) ─────────────
        elif stage == "agent_rewrite_ready":
            agent_id      = update.get("agent_id", "")
            strategy_name = update.get("strategy", "")
            if agent_id and agent_id not in st.session_state.cur_agent_ids:
                st.session_state.cur_agent_ids.append(agent_id)
            st.session_state.cur_agent_id = None
            st.session_state.cur_agent_cards.append({
                "ok":    True, "warn": False,
                "agent_id":              agent_id,
                "strategy_name":         strategy_name,
                "text":                  f"✅ Agent {agent_id} ({strategy_name}) — ready",
                "rewritten_sql":         update.get("rewritten_sql"),
                "optimization_strategy": update.get("optimization_strategy", [])
            })

        # ── Agent failed (LLM or safety) ──────────────────────────
        elif stage == "agent_failed":
            agent_id = update.get("agent_id", "")
            if agent_id and agent_id not in st.session_state.cur_agent_ids:
                st.session_state.cur_agent_ids.append(agent_id)
            st.session_state.cur_agent_id = None
            st.session_state.cur_agent_cards.append({
                "ok":    False, "warn": False,
                "agent_id":      agent_id,
                "strategy_name": update.get("strategy", ""),
                "text":          f"{update['reason']}",
                "rewritten_sql": update.get("rewritten_sql")
            })

        # ── Execution started ─────────────────────────────────────
        elif stage == "execution_started":
            gen       = update["generation"]
            count     = update["executing_count"]
            agent_ids = update.get("executing_agents", [])
            st.session_state.exec_agent_ids  = agent_ids
            st.session_state.exec_agent_id   = None
            status_placeholder.info(
                f"⚡ Gen {gen}/{num_generations} — Executing all {count} agents on Synapse"
            )
            render_execution_panel(gen, num_generations,
                                   st.session_state.exec_agent_ids,
                                   st.session_state.exec_agent_id,
                                   st.session_state.execution_cards)

        # ── Agent executing ───────────────────────────────────────
        elif stage == "agent_executing":
            gen      = update["generation"]
            agent_id = update["agent_id"]
            strategy = update["strategy"]
            st.session_state.exec_agent_id   = agent_id
            st.session_state.exec_start_time = time.time()
            if agent_id not in st.session_state.exec_agent_ids:
                st.session_state.exec_agent_ids.append(agent_id)
            status_placeholder.info(
                f"⚡ Gen {gen}/{num_generations} — "
                f"Agent `{agent_id[:8]}` ({strategy}) running on Synapse..."
            )
            render_execution_panel(gen, num_generations,
                                   st.session_state.exec_agent_ids,
                                   st.session_state.exec_agent_id,
                                   st.session_state.execution_cards)

        # ── Agent executed success ────────────────────────────────
        elif stage == "agent_executed_success":
            aid          = update["agent_id"]
            improvement  = update["improvement"]
            sign         = "+" if improvement >= 0 else ""
            exec_time    = update.get("actual_fitness", 0)
            plan_metrics = update.get("plan_metrics", {})
            strategy     = update.get("strategy", "")
            st.session_state.exec_agent_id = None

            card = {
                "ok":        True,
                "agent_id":  aid,
                "strategy":  strategy,
                "text":      (
                    f"🎯 Agent {aid[:8]} ({strategy}) — "
                    f"⏱️ {exec_time:.2f}s  ({sign}{improvement:.2f}% vs baseline)"
                ),
                "execution_time":        exec_time,
                "improvement":           improvement,
                "plan_metrics":          plan_metrics,
                "rewritten_sql":         update.get("rewritten_sql"),
                "optimization_strategy": update.get("optimization_strategy", []),
            }
            st.session_state.execution_cards.append(card)
            render_execution_panel(update["generation"], num_generations,
                                   st.session_state.exec_agent_ids,
                                   st.session_state.exec_agent_id,
                                   st.session_state.execution_cards)

        # ── Agent validation failed ───────────────────────────────
        elif stage == "agent_validation_failed":
            aid = update["agent_id"]
            st.session_state.exec_agent_id = None
            st.session_state.execution_cards.append({
                "ok": False, "agent_id": aid,
                "strategy": update.get("strategy", ""),
                "text": f"validation failed: {update['reason']}"
            })
            render_execution_panel(update["generation"], num_generations,
                                   st.session_state.exec_agent_ids,
                                   st.session_state.exec_agent_id,
                                   st.session_state.execution_cards)

        # ── Agent execution failed ────────────────────────────────
        elif stage == "agent_execution_failed":
            aid = update["agent_id"]
            st.session_state.exec_agent_id = None
            st.session_state.execution_cards.append({
                "ok": False, "agent_id": aid,
                "strategy": update.get("strategy", ""),
                "text": "execution failed on Synapse"
            })
            render_execution_panel(update["generation"], num_generations,
                                   st.session_state.exec_agent_ids,
                                   st.session_state.exec_agent_id,
                                   st.session_state.execution_cards)

        # ── Generation complete ───────────────────────────────────
        elif stage == "generation_complete":
            gen      = update["generation"]
            best_fit = update["best_fitness"]   # seconds
            impr     = update["improvement"]

            status_placeholder.success(
                f"✅ Gen {gen}/{num_generations} done — Best: {best_fit:.2f}s  ({impr:+.2f}%)"
            )
            st.session_state.fitness_history.append({
                "generation":   gen,
                "label":        f"Gen {gen}",
                "best_fitness": best_fit,
                "improvement":  impr
            })
            st.session_state.failure_patterns = update.get("failure_summary", {})

            # Merge execution card data into agent population for detail view
            exec_cards_by_id = {c["agent_id"]: c for c in st.session_state.execution_cards}
            enriched_cards = []
            for card in st.session_state.cur_agent_cards:
                aid  = card.get("agent_id", "")
                ecrd = exec_cards_by_id.get(aid)
                if ecrd:
                    enriched_cards.append({**card,
                        "plan_metrics":  ecrd.get("plan_metrics", {}),
                        "execution_time": ecrd.get("execution_time"),
                        "ok":            ecrd.get("ok", card.get("ok")),
                    })
                else:
                    enriched_cards.append(card)

            st.session_state.completed_gens.append({
                "generation":        gen,
                "best_fitness":      best_fit,
                "improvement":       impr,
                "population":        update["population"],
                "agent_cards":       enriched_cards,
                "generation_context": st.session_state.generation_context
            })

            with live_placeholder.container():
                sign = "+" if impr >= 0 else ""
                st.success(
                    f"✅ Generation {gen} complete — "
                    f"Best: **{best_fit:.2f}s**  ({sign}{impr:.2f}%)"
                )
                if gen < num_generations:
                    st.caption("▶ Starting next generation...")

            rebuild_history()

        # ── Feedback context built ────────────────────────────────
        elif stage == "generation_context_built":
            st.session_state.generation_context = update.get("generation_context")

        # ── Convergence ───────────────────────────────────────────
        elif stage == "convergence_detected":
            status_placeholder.warning(
                f"⚠️ Converged at Gen {update['generation']} — "
                f"improvement < 2s, stopping early"
            )

        elif stage == "reproduction_complete":
            pass

        # ── Evolution complete ────────────────────────────────────
        elif stage == "evolution_complete":
            status_placeholder.success("🎉 Evolution Complete!")
            st.session_state.is_running = False
            live_placeholder.empty()

            final_time = update["final_fitness"]
            baseline_t = update["baseline_fitness"]
            impr       = update["improvement"]

            with winner_container:
                st.balloons()
                st.header("🏆 Winning Query")
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ Final Time",    f"{final_time:.2f}s",
                          delta=f"{final_time - baseline_t:.2f}s vs baseline")
                c2.metric("📈 Improvement",   f"{impr:+.2f}%")
                c3.metric("🧬 Strategy",       update["winner_strategy"])
                st.markdown("---")
                show_sql_diff(sql_to_run, update.get("winning_sql") or sql_to_run)

            with analytics_container:
                st.header("📊 Evolution Analytics")

                if st.session_state.fitness_history:
                    st.subheader("📉 Execution Time per Generation (lower = faster)")
                    df_h = pd.DataFrame(st.session_state.fitness_history)
                    baseline_val = df_h.loc[df_h["generation"] == 0, "best_fitness"].values[0] \
                                   if 0 in df_h["generation"].values else df_h["best_fitness"].iloc[0]

                    def _bar_color(row):
                        if row["generation"] == 0:
                            return "Baseline"
                        return "Improved" if row["best_fitness"] < baseline_val else "Regressed"

                    df_h["result"] = df_h.apply(_bar_color, axis=1)
                    df_h["label_str"] = df_h.apply(
                        lambda r: f"{r['best_fitness']:.1f}s ({r['improvement']:+.1f}%)"
                                  if r["generation"] > 0 else f"{r['best_fitness']:.1f}s (baseline)",
                        axis=1
                    )

                    try:
                        import altair as alt
                        chart = alt.Chart(df_h).mark_bar().encode(
                            x=alt.X("label:N", sort=None, title="Generation"),
                            y=alt.Y("best_fitness:Q", title="Execution Time (seconds)",
                                    scale=alt.Scale(zero=False)),
                            color=alt.Color("result:N", scale=alt.Scale(
                                domain=["Baseline", "Improved", "Regressed"],
                                range=["#888888", "#2ecc71", "#e74c3c"]
                            )),
                            tooltip=["label:N", "best_fitness:Q", "label_str:N", "result:N"]
                        ).properties(height=300)
                        rule = alt.Chart(pd.DataFrame({"y": [baseline_val]})).mark_rule(
                            color="#888888", strokeDash=[4, 4]
                        ).encode(y="y:Q")
                        st.altair_chart(chart + rule, use_container_width=True)
                        st.caption("Grey dashed = baseline. 🟢 Green = faster. 🔴 Red = slower.")
                    except Exception:
                        st.line_chart(df_h.set_index("label")["best_fitness"])

                if st.session_state.failure_patterns:
                    st.subheader("⚠️ Failure Patterns")
                    fp = st.session_state.failure_patterns
                    fail_df = pd.DataFrame(
                        [{"Failure": k, "Count": v} for k, v in fp.items()]
                    )
                    st.bar_chart(fail_df.set_index("Failure")["Count"])
                    if any("SCHEMA_REMOVED" in k for k in fp):
                        st.warning("⚠️ Agents that removed schema names were penalized")
                    if "ROW_COUNT_MISMATCH" in fp:
                        st.warning("⚠️ Some agents changed query semantics")
