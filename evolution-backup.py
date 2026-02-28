"""
evolution.py  — v3.2

Reproduction: back to v3 style — top-2 elites + (population_size - 2) crossover children.
No random injection. No elite re-execution.

Fitness: execution_time in seconds only (v3.1). Plan cost excluded from fitness.

EXPLAIN: still runs on each freshly-rewritten agent for two purposes only:
  Display only — plan shape (shuffle_ops, broadcast_ops, scans, cost) stored as plan_metrics after execution

Elites: skip LLM rewrite and skip execution. Carry their SQL and fitness forward intact.
This prevents elite degradation from Synapse variance (seen in log: 108s elite re-ran as 153s).

Stage flow per generation:
  generation_started
    → agent_rewriting        (fresh agents only — elites skipped)
    → agent_rewrite_ready    (LLM + safety passed)
    → agent_failed           (LLM error or safety rejection)
  execution_started
    → agent_executing
    → agent_executed_success (includes execution_time + plan_metrics)
    → agent_execution_failed
    → agent_validation_failed
  generation_complete
  generation_context_built
  reproduction_complete
"""

import difflib
import random
from typing import List, Dict, Any, Generator, Optional
from collections import defaultdict

from genome import Genome
from agent import StrategyAgent
from fitness import FitnessEvaluator
from validator import QueryValidator
from safety import SafetyGovernor


class EvolutionEngine:

    def __init__(
        self,
        synapse_client,
        openai_client,
        model_name: str,
        population_size: int = 5,
        generations: int = 3,
        convergence_patience: int = 3
    ):
        self.synapse = synapse_client
        self.openai_client = openai_client
        self.model_name = model_name
        self.population_size = population_size
        self.generations = generations
        self.convergence_patience = convergence_patience

        self.fitness_evaluator = FitnessEvaluator()
        self.validator = QueryValidator()
        self.safety = SafetyGovernor()
        self.failure_patterns = defaultdict(int)

    def initialize_population(self, generation: int) -> List[StrategyAgent]:
        return [StrategyAgent(Genome.random(), generation) for _ in range(self.population_size)]

    def _fetch_known_schemas(self) -> List[str]:
        try:
            rows, _ = self.synapse.execute_query("SELECT name FROM sys.schemas")
            return [row[0] for row in rows]
        except Exception as e:
            print(f"Warning: Could not fetch schema list: {e}")
            return []

    # --------------------------------------------------
    # Feedback loop helpers
    # --------------------------------------------------
    @staticmethod
    def _summarize_diff(original_sql: str, rewritten_sql: str) -> str:
        if not rewritten_sql:
            return ""
        diff = list(difflib.unified_diff(
            original_sql.splitlines(),
            rewritten_sql.splitlines(),
            lineterm=""
        ))
        added   = [l[1:].strip() for l in diff if l.startswith('+') and not l.startswith('+++')]
        removed = [l[1:].strip() for l in diff if l.startswith('-') and not l.startswith('---')]
        added   = [l for l in added   if l][:4]
        removed = [l for l in removed if l][:4]
        parts = []
        if removed:
            parts.append("Removed:\n" + "\n".join(f"  - {l}" for l in removed))
        if added:
            parts.append("Added:\n"   + "\n".join(f"  + {l}" for l in added))
        return "\n".join(parts)

    def _build_generation_context(
        self,
        population: List[StrategyAgent],
        original_sql: str,
        baseline_fitness: float
    ) -> Optional[Dict[str, Any]]:
        executed = [
            a for a in population
            if a.actual_fitness is not None and a.actual_fitness < 9999.0
        ]
        if not executed:
            return None

        best_agent = min(executed, key=lambda a: a.actual_fitness)
        improvement = self.fitness_evaluator.improvement_percentage(
            baseline_fitness, best_agent.actual_fitness
        )

        failed_strategies = []
        for a in population:
            if a.failure_reason:
                failed_strategies.append(
                    f"{a.genome.get_dominant_strategy()} ({a.failure_reason})"
                )
            elif a.actual_fitness is not None and a.actual_fitness > baseline_fitness * 1.05:
                failed_strategies.append(
                    f"{a.genome.get_dominant_strategy()} (made query slower: {a.actual_fitness:.1f}s)"
                )

        diff_summary = self._summarize_diff(original_sql, best_agent.rewritten_sql or "")
        # Preserve TOP N in feedback SQL if original query used it
        if original_has_top_n:
            clean_best_sql = best_agent.rewritten_sql or ""
        else:
            clean_best_sql = self.synapse.strip_top_n(best_agent.rewritten_sql or "")

        return {
            "winner": {
                "strategy":       best_agent.genome.get_dominant_strategy(),
                "improvement":    improvement,
                "execution_time": f"{best_agent.actual_fitness:.2f}",
                "diff_summary":   diff_summary,
            },
            "best_sql":          clean_best_sql,
            "failed_strategies": list(set(failed_strategies)),
        }

    # --------------------------------------------------
    # Main evolution loop
    # --------------------------------------------------
    def run(self, original_sql: str, schema_metadata: str) -> Generator[Dict[str, Any], None, None]:

        # ---------- Safety ----------
        # Detect if the original query legitimately uses TOP N.
        # If so, strip_top_n() must NOT run on rewrites — the LLM should preserve it.
        import re as _re
        original_has_top_n = bool(_re.search(r'(?i)SELECT\s+TOP\s+[\d(]', original_sql))

        is_safe, safety_reason = self.safety.validate(original_sql)
        if not is_safe:
            yield {"stage": "fatal_error", "reason": safety_reason,
                   "message": f"Input query rejected: {safety_reason}"}
            return

        known_schemas    = self._fetch_known_schemas()
        required_schemas = self.safety.extract_schemas(original_sql, known_schemas)

        # ---------- Baseline ----------
        yield {"stage": "baseline_started"}
        try:
            baseline_rows, baseline_metrics = self.synapse.execute_query(original_sql)
        except Exception as e:
            yield {"stage": "fatal_error", "reason": "BASELINE_EXECUTION_FAILED", "message": str(e)}
            return

        baseline_checksum = self.synapse.compute_checksum(baseline_rows)

        # EXPLAIN baseline for display
        try:
            baseline_plan = self.synapse.explain_query(original_sql)
        except Exception:
            baseline_plan = {}
        baseline_metrics.update(baseline_plan)

        baseline_fitness = self.fitness_evaluator.compute(baseline_metrics)

        yield {
            "stage":            "baseline_complete",
            "baseline_metrics": baseline_metrics,
            "baseline_fitness": baseline_fitness,   # seconds
            "baseline_plan":    baseline_plan,
            "baseline_sql":     original_sql
        }

        population  = self.initialize_population(generation=1)
        best_fitness = baseline_fitness
        stagnant_generations = 0
        generation_context: Optional[Dict[str, Any]] = None

        # Mark agents that are elites from a previous gen (should skip rewrite + execution)
        # Initially all agents are fresh
        elite_ids: set = set()

        # ---------- Generations ----------
        for gen in range(1, self.generations + 1):
            yield {
                "stage":              "generation_started",
                "generation":         gen,
                "population_size":    len(population),
                "generation_context": generation_context
            }

            # ── Phase 1: LLM rewrites (fresh agents only) ──────────
            for idx, agent in enumerate(population):
                strategy            = agent.genome.get_dominant_strategy()
                genome_instructions = agent.genome.to_strategy_instructions()

                # Elites carry their previous SQL and fitness — skip rewrite
                if agent.id in elite_ids:
                    continue

                yield {
                    "stage":               "agent_rewriting",
                    "generation":          gen,
                    "agent_id":            agent.id,
                    "agent_index":         idx + 1,
                    "total_agents":        len(population),
                    "strategy":            strategy,
                    "genome_instructions": genome_instructions,
                    "genome_values":       agent.genome.to_dict(),
                    "one_liner":           agent.genome.to_one_liner()
                }

                success = agent.generate_rewrite(
                    original_sql, schema_metadata, baseline_metrics,
                    self.openai_client, self.model_name,
                    required_schemas, generation_context
                )

                if not success:
                    agent.set_actual_fitness(self.fitness_evaluator.penalize())
                    self.failure_patterns[agent.failure_reason] += 1
                    yield {"stage": "agent_failed", "generation": gen,
                           "agent_id": agent.id, "reason": agent.failure_reason,
                           "strategy": strategy}
                    continue

                # Strip LLM-hallucinated TOP N — but only if the original query
                # didn't legitimately use TOP N (we don't want to remove it in that case)
                if not original_has_top_n:
                    agent.rewritten_sql = self.synapse.strip_top_n(agent.rewritten_sql)

                # Safety check
                is_safe, reason = self.safety.validate(agent.rewritten_sql, required_schemas)
                if not is_safe:
                    agent.failure_reason = reason
                    agent.set_actual_fitness(self.fitness_evaluator.penalize())
                    self.failure_patterns[reason] += 1
                    yield {"stage": "agent_failed", "generation": gen,
                           "agent_id": agent.id, "reason": reason,
                           "strategy": strategy, "rewritten_sql": agent.rewritten_sql}
                    continue

                # Agent passed LLM + safety — mark as ready for execution
                agent.explain_metrics = {}   # populated after execution (display only)

                yield {
                    "stage":                 "agent_rewrite_ready",
                    "generation":            gen,
                    "agent_id":              agent.id,
                    "strategy":              strategy,
                    "one_liner":             agent.genome.to_one_liner(),
                    "genome_values":         agent.genome.to_dict(),
                    "rewritten_sql":         agent.rewritten_sql,
                    "optimization_strategy": agent.optimization_strategy
                }

            # ── Phase 2: Execute ALL valid agents ──────────────────
            # No pre-selection — execution time is the only real signal
            # Elites already have actual_fitness — skip them
            execution_group = [
                a for a in population
                if a.actual_fitness is None
                and a.failure_reason is None
                and a.id not in elite_ids
            ]

            yield {
                "stage":            "execution_started",
                "generation":       gen,
                "executing_count":  len(execution_group),
                "executing_agents": [a.id for a in execution_group],
                "total_agents":     len(population)
            }

            for agent in execution_group:
                yield {
                    "stage":      "agent_executing",
                    "generation": gen,
                    "agent_id":   agent.id,
                    "strategy":   agent.genome.get_dominant_strategy()
                }

                result = self.synapse.safe_execute(agent.rewritten_sql)
                if not result:
                    agent.set_actual_fitness(self.fitness_evaluator.penalize())
                    agent.failure_reason = "EXECUTION_FAILED"
                    self.failure_patterns["EXECUTION_FAILED"] += 1
                    yield {"stage": "agent_execution_failed", "generation": gen,
                           "agent_id": agent.id,
                           "strategy": agent.genome.get_dominant_strategy()}
                    continue

                rows, metrics = result
                checksum = self.synapse.compute_checksum(rows)
                valid, reason = self.validator.validate(
                    baseline_rows, rows, baseline_checksum, checksum
                )
                if not valid:
                    agent.failure_reason = reason
                    agent.set_actual_fitness(self.fitness_evaluator.penalize())
                    self.failure_patterns[reason] += 1
                    yield {"stage": "agent_validation_failed", "generation": gen,
                           "agent_id": agent.id, "reason": reason,
                           "strategy": agent.genome.get_dominant_strategy()}
                    continue

                agent.actual_metrics = metrics

                # EXPLAIN after execution — display only, never affects fitness
                try:
                    agent.plan_metrics = self.synapse.explain_query(agent.rewritten_sql)
                except Exception:
                    agent.plan_metrics = {}

                # Fitness = execution time in seconds only
                actual_fitness = self.fitness_evaluator.compute(metrics)
                agent.set_actual_fitness(actual_fitness)
                improvement = self.fitness_evaluator.improvement_percentage(
                    baseline_fitness, actual_fitness
                )

                yield {
                    "stage":                 "agent_executed_success",
                    "generation":            gen,
                    "agent_id":              agent.id,
                    "strategy":              agent.genome.get_dominant_strategy(),
                    "one_liner":             agent.genome.to_one_liner(),
                    "genome_values":         agent.genome.to_dict(),
                    "actual_fitness":        actual_fitness,
                    "improvement":           improvement,
                    "metrics":               metrics,
                    "plan_metrics":          agent.plan_metrics,
                    "rewritten_sql":         agent.rewritten_sql,
                    "optimization_strategy": agent.optimization_strategy,
                }

            # ── Generation summary ─────────────────────────────────
            executed = [a for a in population if a.actual_fitness is not None and a.actual_fitness < 9999.0]
            if executed:
                current_best = min(a.actual_fitness for a in executed)
            else:
                current_best = best_fitness

            improvement = self.fitness_evaluator.improvement_percentage(baseline_fitness, current_best)

            population_summary = []
            for a in population:
                fitness  = a.actual_fitness if a.actual_fitness is not None else float("inf")
                executed = a.actual_fitness is not None and a.actual_fitness < 9999.0
                is_elite = a.id in elite_ids

                if is_elite:
                    exec_reason = "Elite — carried from previous generation"
                elif executed:
                    exec_reason = "Executed (real fitness)"
                else:
                    exec_reason = "Not executed (LLM/safety failed)"

                population_summary.append({
                    "agent_id":           a.id,
                    "strategy":           a.genome.get_dominant_strategy(),
                    "fitness":            fitness,
                    "status":             a.status,
                    "failure_reason":     a.failure_reason,
                    "schema_removed":     a.schema_removed,
                    "has_actual_fitness": executed,
                    "execution_reason":   exec_reason,
                    "plan_metrics":       getattr(a, "plan_metrics", {}) or {},
                })

            yield {
                "stage":          "generation_complete",
                "generation":     gen,
                "best_fitness":   current_best,
                "improvement":    improvement,
                "population":     population_summary,
                "failure_summary": dict(self.failure_patterns)
            }

            # ── Build feedback context for next gen ────────────────
            generation_context = self._build_generation_context(
                population, original_sql, baseline_fitness
            )
            if generation_context:
                yield {
                    "stage":              "generation_context_built",
                    "generation":         gen,
                    "generation_context": generation_context
                }

            # ── Convergence ────────────────────────────────────────
            if abs(best_fitness - current_best) < 0.01:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
                best_fitness = current_best

            if stagnant_generations >= self.convergence_patience:
                yield {"stage": "convergence_detected", "generation": gen}
                break

            # ── Reproduction: top-2 elites + crossover children ───
            all_agents = sorted(population, key=lambda a: a.get_fitness())
            parent1 = all_agents[0]
            parent2 = all_agents[1] if len(all_agents) > 1 else all_agents[0]

            new_population = []
            for _ in range(self.population_size - 2):
                child_genome = Genome.crossover(parent1.genome, parent2.genome)
                child_genome = child_genome.mutate(mutation_rate=0.1)
                new_population.append(
                    StrategyAgent(child_genome, generation=gen + 1,
                                  parent_ids=[parent1.id, parent2.id])
                )

            # Carry elites forward — mark them so next gen skips their rewrite+execution
            new_population.extend([parent1, parent2])
            elite_ids = {parent1.id, parent2.id}

            population = new_population
            yield {"stage": "reproduction_complete", "generation": gen,
                   "new_agents": len(new_population)}

        # ---------- Final winner ----------
        all_agents = sorted(population, key=lambda a: a.get_fitness())
        winner = all_agents[0]
        final_improvement = self.fitness_evaluator.improvement_percentage(
            baseline_fitness, winner.get_fitness()
        )

        yield {
            "stage":                 "evolution_complete",
            "winner_id":             winner.id,
            "winner_strategy":       winner.genome.get_dominant_strategy(),
            "final_fitness":         winner.get_fitness(),
            "winning_sql":           winner.rewritten_sql,
            "improvement":           final_improvement,
            "baseline_fitness":      baseline_fitness,
            "generations_completed": gen,
            "failure_patterns":      dict(self.failure_patterns)
        }