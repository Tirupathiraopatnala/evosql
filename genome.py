"""
genome.py  — v3
Changes from original:
  - Specialist initialization: one trait always forced high (0.75-1.0), others low (0-0.35)
  - Safety net in __init__: impossible to create all-zero genome from any code path
  - Dominant-trait crossover: child inherits one parent's strongest trait intact (not blended)
  - Protected mutation: dominant trait never drops below 0.5, 10% exploration jump
"""

import random
from copy import deepcopy
from typing import List, Dict

from debug_logger import get_logger

log = get_logger("genome")


class Genome:
    def __init__(
        self,
        predicate_pushdown_bias:          float,
        shuffle_avoidance_bias:           float,
        broadcast_tolerance:              float,
        join_reorder_weight:              float,
        temp_table_materialization_bias:  float,
        aggregation_pushdown_bias:        float,
        index_exploitation_bias:          float,
        partition_elimination_bias:       float
    ):
        self.predicate_pushdown_bias         = predicate_pushdown_bias
        self.shuffle_avoidance_bias          = shuffle_avoidance_bias
        self.broadcast_tolerance             = broadcast_tolerance
        self.join_reorder_weight             = join_reorder_weight
        self.temp_table_materialization_bias = temp_table_materialization_bias
        self.aggregation_pushdown_bias       = aggregation_pushdown_bias
        self.index_exploitation_bias         = index_exploitation_bias
        self.partition_elimination_bias      = partition_elimination_bias

        # Safety net: if ALL traits are below 0.4, force the highest one to 0.85
        # This catches any crossover/mutation edge case that produces all-low genomes
        all_values = [
            self.predicate_pushdown_bias, self.shuffle_avoidance_bias,
            self.broadcast_tolerance, self.join_reorder_weight,
            self.temp_table_materialization_bias, self.aggregation_pushdown_bias,
            self.index_exploitation_bias, self.partition_elimination_bias
        ]
        if max(all_values) < 0.4:
            dominant_attr = max(self.__dict__.keys(), key=lambda t: getattr(self, t))
            setattr(self, dominant_attr, 0.85)
            log.warning(
                f"Genome.__init__ safety net triggered — "
                f"all values were < 0.4, forced {dominant_attr}=0.85"
            )

    # --------------------------------------------------
    @staticmethod
    def random() -> 'Genome':
        """
        Specialist genome: one randomly chosen trait is forced high (0.75-1.0),
        all others stay low (0.0-0.35). Every agent has a clear identity from birth.
        Prevents all-zero genomes that produce the fallback 'general best practices' prompt.
        """
        traits = [
            "predicate_pushdown_bias",
            "shuffle_avoidance_bias",
            "broadcast_tolerance",
            "join_reorder_weight",
            "temp_table_materialization_bias",
            "aggregation_pushdown_bias",
            "index_exploitation_bias",
            "partition_elimination_bias",
        ]
        values = {t: random.uniform(0.0, 0.35) for t in traits}
        specialist = random.choice(traits)
        values[specialist] = random.uniform(0.75, 1.0)

        g = Genome(**values)
        log.debug(f"Genome.random() → specialist={specialist}  values={g.to_dict()}")
        return g

    # --------------------------------------------------
    @staticmethod
    def crossover(parent1: 'Genome', parent2: 'Genome') -> 'Genome':
        """
        Dominant-trait crossover:
        - Child randomly inherits the dominant trait from either parent1 or parent2, intact
        - All other traits are blended normally
        Prevents two specialists from producing an all-medium generalist child.
        """
        traits = list(parent1.__dict__.keys())

        p1_dominant = max(traits, key=lambda t: getattr(parent1, t))
        p2_dominant = max(traits, key=lambda t: getattr(parent2, t))
        inherited_dominant = p1_dominant if random.random() < 0.5 else p2_dominant
        dominant_value = max(
            getattr(parent1, inherited_dominant),
            getattr(parent2, inherited_dominant)
        )

        child_dict = {}
        for attr in traits:
            if attr == inherited_dominant:
                child_dict[attr] = dominant_value
            else:
                w = random.random()
                child_dict[attr] = getattr(parent1, attr) * w + getattr(parent2, attr) * (1 - w)

        child = Genome(**child_dict)
        log.debug(
            f"Genome.crossover() → dominant={child.get_dominant_strategy()}  "
            f"inherited={inherited_dominant}={dominant_value:.3f}  "
            f"p1={parent1.get_dominant_strategy()}  p2={parent2.get_dominant_strategy()}"
        )
        return child

    # --------------------------------------------------
    def mutate(self, mutation_rate: float = 0.2) -> 'Genome':
        """
        Protected mutation:
        - Dominant trait: small perturbation only (±0.1), never drops below 0.5
        - Other traits: normal mutation, capped at 0.5 so they don't accidentally dominate
        - 10% chance of exploration jump: randomly boosts a non-dominant trait high
        """
        new_genome = deepcopy(self)
        changes = {}

        dominant_attr = max(self.__dict__.keys(), key=lambda t: getattr(self, t))

        exploration_attr = None
        if random.random() < 0.10:
            non_dominant = [a for a in self.__dict__ if a != dominant_attr]
            exploration_attr = random.choice(non_dominant)

        for attr in self.__dict__:
            old_val = getattr(self, attr)
            if attr == exploration_attr:
                new_val = random.uniform(0.7, 1.0)
            elif attr == dominant_attr:
                delta   = random.uniform(-0.1, 0.1)
                new_val = max(0.5, min(1.0, old_val + delta))
            else:
                delta   = random.uniform(-mutation_rate, mutation_rate)
                new_val = max(0.0, min(0.5, old_val + delta))
            setattr(new_genome, attr, new_val)
            if abs(new_val - old_val) > 0.05:
                changes[attr] = f"{old_val:.3f} → {new_val:.3f}"

        log.debug(f"Genome.mutate() changes: {changes}  exploration={exploration_attr}")
        log.debug(f"  new dominant={new_genome.get_dominant_strategy()}")
        return new_genome

    # --------------------------------------------------
    def to_strategy_instructions(self) -> List[str]:
        instructions = []

        checks = [
            (self.predicate_pushdown_bias,         0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Aggressively move ALL WHERE clauses before JOINs to filter rows early",
             "⚖️ MODERATE: Consider moving WHERE clauses before JOINs when beneficial"),

            (self.shuffle_avoidance_bias,           0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Minimize data movement — avoid unnecessary shuffles, align joins with distribution keys",
             "⚖️ MODERATE: Try to reduce data shuffles where possible"),

            (self.join_reorder_weight,              0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Reorder joins to put smallest table first, largest last",
             "⚖️ MODERATE: Consider reordering joins for efficiency"),

            (self.temp_table_materialization_bias,  0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Materialize complex subqueries or CTEs into temp tables with proper distribution",
             "⚖️ MODERATE: Consider temp tables for complex intermediate results"),

            (self.aggregation_pushdown_bias,        0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Push aggregations (GROUP BY, SUM, COUNT) as close to base tables as possible",
             "⚖️ MODERATE: Look for opportunities to aggregate earlier"),

            (self.index_exploitation_bias,          0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Rewrite predicates to use existing indexes — avoid functions on indexed columns",
             "⚖️ MODERATE: Try to leverage available indexes"),

            (self.partition_elimination_bias,       0.7, 0.4,
             "🔥 CRITICAL PRIORITY: Add partition column filters to enable partition elimination",
             "⚖️ MODERATE: Look for partition elimination opportunities"),
        ]

        for value, high_thresh, mid_thresh, high_text, mid_text in checks:
            if value > high_thresh:
                instructions.append(high_text)
            elif value > mid_thresh:
                instructions.append(mid_text)

        if self.broadcast_tolerance > 0.7:
            instructions.append("✅ ALLOWED: Use REPLICATE hints for small dimension tables (< 10GB)")
        elif self.broadcast_tolerance < 0.3:
            instructions.append("🚫 AVOID: Broadcasting tables — minimize replicate operations")

        if not instructions:
            log.warning(
                f"to_strategy_instructions() → EMPTY (all biases < 0.4). "
                f"Values: {self.to_dict()}"
            )

        return instructions

    # --------------------------------------------------
    def get_dominant_strategy(self) -> str:
        traits = {
            "Predicate Pusher":      self.predicate_pushdown_bias,
            "Shuffle Avoider":       self.shuffle_avoidance_bias,
            "Broadcast User":        self.broadcast_tolerance,
            "Join Optimizer":        self.join_reorder_weight,
            "Temp Table Creator":    self.temp_table_materialization_bias,
            "Aggregation Pusher":    self.aggregation_pushdown_bias,
            "Index Exploiter":       self.index_exploitation_bias,
            "Partition Eliminator":  self.partition_elimination_bias
        }
        return max(traits.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}

    def __repr__(self):
        return f"Genome({self.get_dominant_strategy()}: {self.to_dict()})"
