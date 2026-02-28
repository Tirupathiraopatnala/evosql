"""
agent.py  — v3.1

Changes from v3:
  - Added plan_metrics field: stores post-execution EXPLAIN data (display only)
  - Removed estimated_metrics / estimated_fitness (no longer used for selection)
  - generate_rewrite() unchanged — still accepts generation_context for feedback loop
  - get_fitness() returns actual_fitness (seconds) or penalty
"""

import uuid
import time
from typing import Optional, Dict, Any, List

from genome import Genome
from debug_logger import get_logger

log = get_logger("agent")


class StrategyAgent:
    def __init__(self, genome: Genome, generation: int, parent_ids: List[str] = None):
        self.id            = str(uuid.uuid4())[:8]
        self.genome        = genome
        self.generation    = generation
        self.parent_ids    = parent_ids or []

        self.rewritten_sql: Optional[str]       = None
        self.optimization_strategy: List[str]   = []

        # Plan shape from post-execution EXPLAIN — display only, does not affect fitness
        self.plan_metrics:   Optional[Dict]  = None

        # Actual execution metrics and fitness (seconds)
        self.actual_metrics: Optional[Dict]  = None
        self.actual_fitness: Optional[float] = None

        self.status:         str            = "INITIALIZED"
        self.failure_reason: Optional[str]  = None
        self.schema_removed: bool           = False

        log.debug(
            f"Agent {self.id} created | "
            f"gen={generation} | "
            f"strategy={genome.get_dominant_strategy()} | "
            f"parents={self.parent_ids or 'none (random)'}"
        )

    # --------------------------------------------------
    def generate_rewrite(
        self,
        original_sql: str,
        schema_metadata: str,
        baseline_metrics: Dict[str, Any],
        openai_client,
        model_name: str,
        required_schemas: List[str] = None,
        generation_context: Optional[Dict[str, Any]] = None
    ) -> bool:

        log.info(f"[Agent {self.id}] generate_rewrite() START — strategy={self.genome.get_dominant_strategy()}")
        self.status = "LLM_RUNNING"
        self.optimization_strategy = self.genome.to_strategy_instructions()

        log.debug(f"[Agent {self.id}] Genome values: {self.genome.to_dict()}")
        log.debug(f"[Agent {self.id}] Instructions ({len(self.optimization_strategy)} items):")
        for i, inst in enumerate(self.optimization_strategy, 1):
            log.debug(f"  {i}. {inst}")

        if not self.optimization_strategy:
            log.warning(f"[Agent {self.id}] ALL genome values < 0.4 — fallback strategy will be used")

        if generation_context:
            log.info(f"[Agent {self.id}] generation_context provided — feedback loop active")

        try:
            prompt = self._build_prompt(
                original_sql, schema_metadata, baseline_metrics,
                required_schemas, generation_context
            )
            log.debug(f"[Agent {self.id}] Prompt built ({len(prompt)} chars)")
            log.debug(f"[Agent {self.id}] Full prompt:\n{'='*60}\n{prompt}\n{'='*60}")

            temperature = 0.4 if generation_context else 0.3
            log.info(f"[Agent {self.id}] Calling OpenAI (model={model_name}, temp={temperature}) …")
            t0 = time.time()

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert Azure Synapse SQL optimizer."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=temperature
            )

            llm_time = time.time() - t0
            log.info(f"[Agent {self.id}] LLM responded in {llm_time:.2f}s")

            raw_sql = response.choices[0].message.content.strip()
            log.debug(f"[Agent {self.id}] Raw response ({len(raw_sql)} chars):\n{raw_sql}")

            if raw_sql.startswith("```"):
                raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

            self.rewritten_sql = raw_sql
            log.debug(f"[Agent {self.id}] Final SQL:\n{self.rewritten_sql}")

            if required_schemas:
                self.schema_removed = self._check_schema_removal(raw_sql, required_schemas)
                if self.schema_removed:
                    log.warning(f"[Agent {self.id}] ⚠️  SCHEMA REMOVAL DETECTED")

            self.status = "LLM_SUCCESS"
            log.info(f"[Agent {self.id}] generate_rewrite() SUCCESS")
            return True

        except Exception as e:
            self.status = "LLM_FAILED"
            self.failure_reason = f"LLM_ERROR: {str(e)}"
            log.error(f"[Agent {self.id}] generate_rewrite() FAILED — {e}")
            return False

    # --------------------------------------------------
    def _check_schema_removal(self, sql: str, required_schemas: List[str]) -> bool:
        sql_upper = sql.upper()
        for schema in required_schemas:
            if schema.upper() not in sql_upper:
                return True
        return False

    # --------------------------------------------------
    def _build_prompt(
        self,
        original_sql: str,
        schema_metadata: str,
        baseline_metrics: Dict[str, Any],
        required_schemas: List[str] = None,
        generation_context: Optional[Dict[str, Any]] = None
    ) -> str:

        strategy_instructions = self.genome.to_strategy_instructions()
        if not strategy_instructions:
            strategy_instructions = [
                "Apply general SQL best practices: predicate pushdown, efficient joins, reduce data movement"
            ]
            log.warning(f"[Agent {self.id}] Using fallback strategy")

        instruction_text = "\n".join(
            f"{i+1}. {inst}" for i, inst in enumerate(strategy_instructions)
        )

        schema_warning = ""
        if required_schemas:
            schema_warning = f"\n⚠️  MANDATORY: These schema prefixes MUST appear in your output: {required_schemas}"

        feedback_section = ""
        if generation_context:
            winner   = generation_context.get("winner", {})
            best_sql = generation_context.get("best_sql", "")
            failed   = generation_context.get("failed_strategies", [])

            lines = ["\n" + "="*60, "LEARNING FROM PREVIOUS GENERATION:"]

            if winner:
                impr = winner.get("improvement", 0)
                lines.append(
                    f"✅ Best approach last gen: '{winner.get('strategy', '?')}' "
                    f"— achieved {impr:+.2f}% improvement ({winner.get('execution_time', 'N/A')}s)"
                )
                diff_summary = winner.get("diff_summary", "")
                if diff_summary:
                    lines.append("What it changed (build on this, don't just copy it):")
                    for line in diff_summary.splitlines():
                        lines.append(f"  {line}")

            if failed:
                lines.append("❌ These approaches did NOT help — avoid repeating them:")
                for f in failed:
                    lines.append(f"  - {f}")

            if best_sql:
                lines.append("\nBEST SQL FROM PREVIOUS GENERATION (improve on this):")
                lines.append(best_sql)

            lines.append("="*60)
            feedback_section = "\n".join(lines)

        baseline_time = baseline_metrics.get("execution_time", "N/A")

        prompt = f"""You are optimizing a slow Azure Synapse SQL query.

ORIGINAL QUERY:
{original_sql}

BASELINE PERFORMANCE:
- Execution Time  : {baseline_time} seconds
- Broadcast Ops   : {baseline_metrics.get('broadcast_ops', 'N/A')}
- Full Table Scans: {baseline_metrics.get('full_scans', 'N/A')}
- Shuffle Ops     : {baseline_metrics.get('shuffle_ops', 'N/A')}

SCHEMA METADATA (DO NOT REMOVE SCHEMA NAMES):
{schema_metadata}
{schema_warning}
{feedback_section}

YOUR OPTIMIZATION STRATEGY (Apply in priority order):
{instruction_text}

CRITICAL RULES:
1. ⚠️  PRESERVE ALL SCHEMA NAMES — never remove schema prefixes from table names
2. ⚠️  PRESERVE CROSS APPLY — never convert CROSS APPLY to CROSS JOIN (they are different operations)
3. ⚠️  PRESERVE CTEs — keep all WITH (...) AS (...) blocks intact, only optimize inside them
4. ⚠️  PRESERVE TOP N — the original query uses " + top_n_clause + ", your rewrite MUST also include " + top_n_clause + " on the final SELECT" if top_n_clause else "Do NOT add TOP N unless it was in the original query
5. Output ONLY valid SQL — no explanations, no markdown, no comments
6. Maintain exact result semantics — same rows, same columns, same order

REWRITE THE QUERY NOW:
"""
        return prompt

    # --------------------------------------------------
    def set_actual_fitness(self, fitness: float):
        self.actual_fitness = fitness
        log.debug(f"[Agent {self.id}] actual_fitness = {fitness:.3f}s")

    def get_fitness(self) -> float:
        if self.actual_fitness is not None:
            return self.actual_fitness
        return float("inf")

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            "agent_id":              self.id,
            "generation":            self.generation,
            "strategy":              self.genome.get_dominant_strategy(),
            "fitness":               self.get_fitness(),
            "status":                self.status,
            "failure_reason":        self.failure_reason,
            "schema_removed":        self.schema_removed,
            "has_actual_fitness":    self.actual_fitness is not None,
            "optimization_strategy": self.optimization_strategy,
            "rewritten_sql":         self.rewritten_sql,
            "plan_metrics":          self.plan_metrics,
        }

    def __repr__(self):
        fitness_str = f"{self.actual_fitness:.2f}s" if self.actual_fitness is not None else "pending"
        return (
            f"<Agent {self.id} | "
            f"Gen {self.generation} | "
            f"{self.genome.get_dominant_strategy()} | "
            f"{fitness_str} | "
            f"{self.status}>"
        )
