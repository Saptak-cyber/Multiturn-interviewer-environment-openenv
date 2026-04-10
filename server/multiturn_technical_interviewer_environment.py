# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-turn Technical Interviewer Environment.

Simulates a realistic technical interview with three tasks of increasing
difficulty. The environment:
  1. Presents a technical problem to the agent (on reset).
  2. Evaluates the agent's solution.
  3. Asks structured follow-up questions covering complexity analysis,
     edge cases, concurrency, and distributed systems design.
  4. Grades each response using deterministic keyword matching.

Tasks
-----
two_sum      (easy,   5 turns) – algorithm + optimisation + distribution
lru_cache    (medium, 6 turns) – data-structure design + concurrency + scale
rate_limiter (hard,   7 turns) – distributed systems design end-to-end

Reward
------
Each turn returns a reward in [0.0, 1.0] based on answer quality:
  • 0.0          : empty / dismissive response
  • 0.1 – 0.3   : attempted but very thin / off-topic
  • 0.3 – 0.6   : partial credit – relevant concepts present
  • 0.6 – 0.8   : solid answer, most required terms covered
  • 0.8 – 1.0   : excellent depth, bonus topics included

Episode score = mean reward across all turns (clamped to [0, 1]).
"""

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        MultiturnTechnicalInterviewerAction,
        MultiturnTechnicalInterviewerObservation,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        MultiturnTechnicalInterviewerAction,
        MultiturnTechnicalInterviewerObservation,
    )


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def _grade_response(
    response: str,
    required_groups: List[List[str]],
    bonus_keywords: List[str],
    min_length: int = 30,
) -> float:
    """
    Return a score in [0.0, 1.0] for *response* using keyword matching.

    Parameters
    ----------
    required_groups:
        List of synonym groups. Each group contributes 1 point when ANY
        keyword in the group appears in the lowercased response.
        Score_required = matched_groups / total_groups.
    bonus_keywords:
        Additional keywords. Each match adds a small bonus.
        Score_bonus = matched_bonus / total_bonus (capped at 1).
    min_length:
        Minimum character count for a non-trivially short answer.
        Shorter answers are penalised proportionally.
    """
    text = (response or "").strip()
    if len(text) < 10:
        return 0.0

    lower = text.lower()

    # Length factor (penalise very short answers)
    length_factor = min(len(text) / max(min_length, 1), 1.0)

    # Required keyword groups
    if required_groups:
        matched = sum(
            1 for grp in required_groups if any(kw in lower for kw in grp)
        )
        req_score = matched / len(required_groups)
    else:
        req_score = 1.0

    # Bonus keywords
    if bonus_keywords:
        bonus_hit = sum(1 for kw in bonus_keywords if kw in lower)
        bonus_score = min(bonus_hit / len(bonus_keywords), 1.0)
    else:
        bonus_score = 0.0

    raw = req_score * 0.75 + bonus_score * 0.25
    score = raw * length_factor

    # Small reward for thorough long-form answers
    if len(text) > 300:
        score = min(score + 0.04, 1.0)

    return round(min(max(score, 0.0), 1.0), 3)


def _build_hint(score: float, task_name: str, turn_idx: int) -> str:
    """Return brief interviewer feedback based on score band."""
    if score >= 0.8:
        return "Good answer — you covered the key points thoroughly."
    if score >= 0.55:
        return "Solid, but consider going deeper on the technical specifics."
    if score >= 0.3:
        return "Partial credit. Try to be more precise about complexities and data structures."
    return "Let me give you a moment to think — try to be more specific and technical."


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------ #
    # TASK 1 – Two Sum (Easy, 5 graded turns)                             #
    # ------------------------------------------------------------------ #
    "two_sum": {
        "display_name": "Two Sum",
        "difficulty": "easy",
        "problem": (
            "**Problem: Two Sum**\n\n"
            "Given an array of integers `nums` and an integer `target`, return the "
            "**indices** of the two numbers that add up to `target`. You may assume "
            "exactly one solution exists; you may not use the same element twice.\n\n"
            "Example:\n"
            "  nums = [2, 7, 11, 15], target = 9  →  [0, 1]\n\n"
            "**Your task:** Write a working solution and explain your approach. "
            "Start with any solution — we will refine it together."
        ),
        "turns": [
            # Turn 1 — Initial solution quality
            {
                "question": (
                    "Walk me through your solution's time complexity. "
                    "The brute-force nested loop is O(n²). "
                    "Can you achieve O(n) and, if so, how?"
                ),
                "required_groups": [
                    ["o(n²)", "n squared", "nested loop", "brute force", "two loop",
                     "o(n)", "linear", "hash", "dictionary", "complement"],
                    ["time complexity", "complexity", "o(n)", "o(n²)", "linear",
                     "quadratic", "hash", "map", "dict"],
                ],
                "bonus": [
                    "hash map", "hash table", "dictionary", "complement",
                    "single pass", "one pass", "o(n)",
                ],
                "min_length": 40,
            },
            # Turn 2 — Time-complexity optimisation
            {
                "question": (
                    "Great. Now discuss the **space complexity** trade-off. "
                    "Your O(n) solution uses extra memory; the O(n²) brute force "
                    "uses O(1) additional space. When would you prefer each approach?"
                ),
                "required_groups": [
                    ["space complexity", "space", "memory", "o(1)", "o(n)"],
                    ["trade-off", "tradeoff", "prefer", "depend", "choose",
                     "when", "advantage", "better", "worse"],
                ],
                "bonus": [
                    "o(1) space", "o(n) space", "memory constrained",
                    "sorted array", "two pointer", "read-only",
                ],
                "min_length": 40,
            },
            # Turn 3 — Space trade-offs
            {
                "question": (
                    "What **edge cases** must your solution handle? "
                    "Consider: empty array, no valid pair, duplicate values, "
                    "and negative numbers."
                ),
                "required_groups": [
                    ["edge case", "empty", "no solution", "no valid", "not found",
                     "duplicate", "negative", "null", "-1", "invalid", "zero"],
                    ["handle", "check", "validate", "guard", "return", "raise",
                     "when", "if"],
                ],
                "bonus": [
                    "empty array", "no valid pair", "duplicate numbers",
                    "negative numbers", "integer overflow", "same index",
                    "return empty", "raise exception",
                ],
                "min_length": 40,
            },
            # Turn 4 — Edge cases
            {
                "question": (
                    "The array now has **10 billion integers** spread across "
                    "100 machines — too large for a single node. "
                    "How do you design a distributed Two Sum solution?"
                ),
                "required_groups": [
                    ["distributed", "partition", "shard", "machine", "node",
                     "server", "across"],
                    ["hash", "complement", "broadcast", "aggregate", "reduce",
                     "coordinator", "map reduce", "lookup"],
                ],
                "bonus": [
                    "consistent hashing", "coordinator node", "broadcast target",
                    "local search", "merge results", "two-phase",
                    "complement lookup", "hash partition",
                ],
                "min_length": 50,
            },
            # Turn 5 — Distributed design
            {
                "question": (
                    "Excellent discussion! Final question: what are the **key "
                    "engineering trade-offs** between the single-machine and "
                    "distributed approaches you described?"
                ),
                "required_groups": [
                    ["trade-off", "tradeoff", "advantage", "disadvantage",
                     "pros", "cons", "better", "worse", "cost"],
                    ["latency", "network", "overhead", "communication",
                     "performance", "scale", "complexity", "fault"],
                ],
                "bonus": [
                    "network overhead", "data transfer", "consistency",
                    "fault tolerance", "simplicity", "operational complexity",
                    "serialize", "bandwidth",
                ],
                "min_length": 40,
            },
        ],
    },

    # ------------------------------------------------------------------ #
    # TASK 2 – LRU Cache (Medium, 6 graded turns)                         #
    # ------------------------------------------------------------------ #
    "lru_cache": {
        "display_name": "LRU Cache",
        "difficulty": "medium",
        "problem": (
            "**Problem: LRU Cache**\n\n"
            "Design a data structure that implements a "
            "**Least Recently Used (LRU) Cache** with:\n"
            "  • `get(key)` → returns the value if present, else -1. "
            "Marks the key as recently used.\n"
            "  • `put(key, value)` → inserts or updates. If at capacity, "
            "evicts the **least recently used** key first.\n\n"
            "**Constraint:** Both `get` and `put` must run in **O(1)** average time.\n\n"
            "Example:\n"
            "  cache = LRUCache(2)\n"
            "  cache.put(1, 1)  # {1:1}\n"
            "  cache.put(2, 2)  # {1:1, 2:2}\n"
            "  cache.get(1)     # → 1   (moves 1 to MRU)\n"
            "  cache.put(3, 3)  # evicts key 2  → {1:1, 3:3}\n"
            "  cache.get(2)     # → -1  (was evicted)\n\n"
            "**Your task:** Describe your data structures and write the implementation."
        ),
        "turns": [
            # Turn 1 — Initial design
            {
                "question": (
                    "Explain **exactly why** your implementation achieves O(1) for "
                    "both `get` and `put`. What role does each data structure play?"
                ),
                "required_groups": [
                    ["o(1)", "constant time", "constant"],
                    ["hash map", "hash table", "dictionary", "dict", "hashmap",
                     "map"],
                    ["linked list", "doubly linked", "dll", "node", "pointer",
                     "list", "deque"],
                ],
                "bonus": [
                    "doubly linked list", "sentinel node", "dummy head",
                    "move to front", "move to head", "hash map for o(1) lookup",
                    "tail is lru",
                ],
                "min_length": 60,
            },
            # Turn 2 — O(1) explanation
            {
                "question": (
                    "Identify all **edge cases** your LRU Cache must handle. "
                    "Think about capacity=1, updating an existing key, and "
                    "calling get on a missing key."
                ),
                "required_groups": [
                    ["edge case", "capacity", "single", "update", "miss",
                     "full", "empty", "invalid", "exist", "present", "not found"],
                    ["handle", "check", "when", "if", "return", "evict",
                     "replace", "guard"],
                ],
                "bonus": [
                    "capacity 1", "key already exists", "update value",
                    "cache miss", "return -1", "evict lru", "capacity zero",
                ],
                "min_length": 50,
            },
            # Turn 3 — Edge cases
            {
                "question": (
                    "Multiple threads concurrently call `get` and `put`. "
                    "What **race conditions** arise, and how do you make your "
                    "implementation thread-safe?"
                ),
                "required_groups": [
                    ["thread", "concurrent", "parallel", "race condition",
                     "synchron", "multi-thread"],
                    ["lock", "mutex", "synchronized", "atomic", "thread-safe",
                     "semaphore", "rwlock", "read-write lock", "reentrant"],
                ],
                "bonus": [
                    "read-write lock", "race condition", "deadlock",
                    "reentrantlock", "synchronized block", "atomic reference",
                    "lock striping", "concurrent hashmap",
                ],
                "min_length": 50,
            },
            # Turn 4 — Concurrency
            {
                "question": (
                    "Scale to a **distributed cache** serving 50 million users. "
                    "How do you partition the cache across multiple servers? "
                    "Which existing system would you use?"
                ),
                "required_groups": [
                    ["distributed", "partition", "shard", "cluster",
                     "multiple server", "node", "scale"],
                    ["consistent hashing", "redis", "memcached",
                     "partition key", "routing", "hash ring", "shard key",
                     "hash slot"],
                ],
                "bonus": [
                    "consistent hashing", "virtual nodes", "replication",
                    "ttl", "eviction policy", "hot keys", "redis cluster",
                    "memcached",
                ],
                "min_length": 60,
            },
            # Turn 5 — Distributed design
            {
                "question": (
                    "Two replicas of your cache diverge — they hold **different "
                    "values** for the same key. How do you handle cache "
                    "invalidation and maintain consistency across replicas?"
                ),
                "required_groups": [
                    ["invalidat", "inconsisten", "stale", "replac", "sync",
                     "expir", "diverge", "conflict"],
                    ["write-through", "write-back", "write-behind", "ttl",
                     "publish", "event", "invalidation", "version",
                     "timestamp", "pub/sub", "eventual consistency"],
                ],
                "bonus": [
                    "write-through", "ttl expiry", "cache invalidation",
                    "pub/sub", "versioning", "eventual consistency",
                    "strong consistency", "cache stampede", "thundering herd",
                ],
                "min_length": 50,
            },
            # Turn 6 — Consistency
            {
                "question": (
                    "Final review: what is the overall **time and space "
                    "complexity** of your distributed design? "
                    "Identify the single biggest **performance bottleneck** "
                    "and how you would address it."
                ),
                "required_groups": [
                    ["time complexity", "space complexity", "o(1)", "o(n)",
                     "complexity"],
                    ["bottleneck", "hotspot", "latency", "network", "slow",
                     "limit", "constraint", "bandwidth"],
                ],
                "bonus": [
                    "network latency", "hot keys", "single point of failure",
                    "consistent hashing", "load balancing", "read replica",
                    "sharding strategy",
                ],
                "min_length": 50,
            },
        ],
    },

    # ------------------------------------------------------------------ #
    # TASK 3 – Distributed Rate Limiter (Hard, 7 graded turns)            #
    # ------------------------------------------------------------------ #
    "rate_limiter": {
        "display_name": "Distributed Rate Limiter",
        "difficulty": "hard",
        "problem": (
            "**Problem: Distributed Rate Limiter**\n\n"
            "Design a **rate limiter** for a REST API that enforces "
            "**100 requests per user per minute**.\n\n"
            "Requirements:\n"
            "  • Start with a correct single-server design.\n"
            "  • Must scale to 50+ API servers sharing state.\n"
            "  • Latency overhead per request ≤ 5 ms.\n"
            "  • Handle failures gracefully — do not take down the API.\n\n"
            "**Your task:** Describe your initial single-server design. "
            "Which rate-limiting algorithm would you choose and why?"
        ),
        "turns": [
            # Turn 1 — Algorithm choice
            {
                "question": (
                    "Compare **token bucket** vs **sliding window counter** "
                    "algorithms in depth. Which handles burst traffic better? "
                    "Walk me through their trade-offs."
                ),
                "required_groups": [
                    ["token bucket"],
                    ["sliding window", "fixed window", "leaky bucket",
                     "counter", "sliding log"],
                    ["burst", "smooth", "trade-off", "tradeoff", "advantage",
                     "disadvantage", "better", "worse", "compare", "differ"],
                ],
                "bonus": [
                    "burst traffic", "refill rate", "smooth traffic",
                    "boundary edge case", "precision", "memory usage",
                    "fixed window boundary",
                ],
                "min_length": 60,
            },
            # Turn 2 — Algorithm comparison
            {
                "question": (
                    "**Implement** the sliding window counter algorithm "
                    "(pseudocode is fine). What is its **time and space "
                    "complexity** per user request?"
                ),
                "required_groups": [
                    ["sliding window", "window", "counter", "timestamp",
                     "queue", "log", "sorted set"],
                    ["time complexity", "space complexity", "o(1)", "o(n)",
                     "per request", "per user", "complexity"],
                ],
                "bonus": [
                    "sorted set", "redis zadd", "zremrangebyscore",
                    "expire", "cleanup", "o(1) amortized",
                    "circular buffer", "redis",
                ],
                "min_length": 60,
            },
            # Turn 3 — Implementation
            {
                "question": (
                    "Your single-server implementation has a **race condition** "
                    "when two concurrent requests arrive simultaneously at the "
                    "rate limit boundary. How do you eliminate it?"
                ),
                "required_groups": [
                    ["race condition", "concurrent", "atomic", "race",
                     "simultaneous", "check-then-act"],
                    ["redis", "lua script", "transaction", "atomic operation",
                     "lock", "mutex", "incr", "pipeline", "multi/exec",
                     "cas", "compare-and-swap"],
                ],
                "bonus": [
                    "lua script", "multi/exec", "atomic increment",
                    "check-then-act", "redis incr", "distributed lock",
                    "optimistic locking", "incrby",
                ],
                "min_length": 50,
            },
            # Turn 4 — Race conditions
            {
                "question": (
                    "Now scale to **50 API servers**. A single user's requests "
                    "hit different servers on each call. How do you enforce a "
                    "**global rate limit** for that user across all servers?"
                ),
                "required_groups": [
                    ["global", "central", "shared", "distributed",
                     "50 server", "multiple server", "centralized",
                     "across server"],
                    ["redis", "database", "shared store", "sync",
                     "communicate", "aggregate", "counter", "central store"],
                ],
                "bonus": [
                    "centralized redis", "sticky session", "token budget",
                    "local + global", "gossip protocol", "sidecar",
                    "service mesh", "approximate counting",
                ],
                "min_length": 60,
            },
            # Turn 5 — Distributed scaling
            {
                "question": (
                    "Your central Redis rate-limit store **crashes**. "
                    "How does the system handle this failure? "
                    "Design for graceful degradation."
                ),
                "required_groups": [
                    ["fail", "crash", "down", "unavailable", "outage"],
                    ["fail open", "fail closed", "fallback", "circuit breaker",
                     "degrade", "local", "in-memory", "allow", "deny",
                     "graceful"],
                ],
                "bonus": [
                    "fail open", "circuit breaker", "local fallback",
                    "retry", "timeout", "health check", "approximate",
                    "fail-safe", "bulkhead",
                ],
                "min_length": 50,
            },
            # Turn 6 — Failure handling
            {
                "question": (
                    "A sophisticated user splits their traffic across **50 "
                    "different client IPs** to evade your per-IP rate limit. "
                    "How do you redesign to prevent this?"
                ),
                "required_groups": [
                    ["ip", "user id", "account", "identity", "limit",
                     "rate limit", "evade", "bypass"],
                    ["user-based", "authentication", "token", "api key",
                     "account id", "fingerprint", "behavior", "oauth",
                     "jwt", "auth"],
                ],
                "bonus": [
                    "user id", "oauth token", "api key",
                    "behavioral analysis", "fingerprinting",
                    "account-level", "device fingerprint",
                    "anomaly detection",
                ],
                "min_length": 50,
            },
            # Turn 7 — Final review
            {
                "question": (
                    "**Final review:** Summarise your complete distributed rate "
                    "limiter design. Cover: algorithm choice, storage layer, "
                    "failure handling, and the single most significant remaining "
                    "limitation."
                ),
                "required_groups": [
                    ["algorithm", "token bucket", "sliding window", "rate limit"],
                    ["redis", "central store", "storage", "database",
                     "shared store"],
                    ["failure", "fail", "graceful", "fallback",
                     "circuit breaker", "degrade"],
                    ["limitation", "trade-off", "tradeoff", "drawback",
                     "challenge", "issue", "concern", "bottleneck"],
                ],
                "bonus": [
                    "scalable", "latency", "consistency", "approximate",
                    "hot keys", "single point of failure", "p99",
                ],
                "min_length": 80,
            },
        ],
    },
}

# Ordered list used for auto-cycling across episodes
TASK_ORDER = ["two_sum", "lru_cache", "rate_limiter"]


# ---------------------------------------------------------------------------
# Module-level episode counter
# ---------------------------------------------------------------------------
# A module-level integer is the most reliable shared state in a single-process
# uvicorn server.  It outlives any number of environment instances and WebSocket
# sessions without needing class-attribute tricks.
#
# The INTERVIEW_TASK env var can pin the first task (default: "two_sum").

def _initial_index() -> int:
    task = os.getenv("INTERVIEW_TASK", "two_sum")
    return TASK_ORDER.index(task) if task in TASK_ORDER else 0


_episode_counter: int = _initial_index()


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class MultiturnTechnicalInterviewerEnvironment(Environment):
    """
    Multi-turn Technical Interviewer OpenEnv environment.

    On each ``reset()`` the environment cycles to the next task (two_sum →
    lru_cache → rate_limiter → two_sum → …), so three consecutive episodes
    cover all three tasks.

    The ``INTERVIEW_TASK`` environment variable can pin the starting task.

    Episode flow
    ------------
    reset()          → returns problem statement (turn 0, reward 0)
    step(solution)   → grades solution, asks follow-up 1   (turn 1, reward r₁)
    step(answer_1)   → grades answer_1, asks follow-up 2   (turn 2, reward r₂)
    …
    step(answer_N)   → grades answer_N, done=True           (turn N, reward rN)

    Episode score = mean(r₁ … rN).

    Implementation note
    -------------------
    Task cycling relies on the module-level ``_episode_counter``.  It is
    shared across all class instances in the same server process regardless of
    how many WebSocket sessions open and close, so consecutive reset() calls
    always advance through the task sequence correctly.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Per-episode mutable state (populated on reset())
        self._task_name: str = TASK_ORDER[0]
        self._task_cfg: Dict[str, Any] = TASKS[TASK_ORDER[0]]
        self._current_turn: int = 0
        self._history: List[str] = []
        self._rewards: List[float] = []
        self._done: bool = False

    # ------------------------------------------------------------------
    def reset(self) -> MultiturnTechnicalInterviewerObservation:
        """Reset the environment and advance to the next task."""
        global _episode_counter
        task_name = TASK_ORDER[_episode_counter % len(TASK_ORDER)]
        _episode_counter += 1

        self._task_name = task_name
        self._task_cfg = TASKS[task_name]
        self._current_turn = 0
        self._history = []
        self._rewards = []
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        problem = self._task_cfg["problem"]
        max_turns = len(self._task_cfg["turns"])

        self._history.append(f"Interviewer: {problem}")

        return MultiturnTechnicalInterviewerObservation(
            question=problem,
            turn=0,
            max_turns=max_turns,
            task_name=task_name,
            task_difficulty=self._task_cfg["difficulty"],
            task_display_name=self._task_cfg["display_name"],
            conversation_history=list(self._history),
            turn_score=None,
            hint="",
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    def step(  # type: ignore[override]
        self,
        action: MultiturnTechnicalInterviewerAction,
    ) -> MultiturnTechnicalInterviewerObservation:
        """
        Evaluate the agent's response and advance to the next interview turn.

        The reward for this step reflects the quality of *action.response*
        against the expected keywords for the current turn.
        """
        if self._done:
            # Episode already finished; return terminal observation with 0 reward
            return MultiturnTechnicalInterviewerObservation(
                question="The interview has already concluded. Call reset() to start a new episode.",
                turn=self._current_turn,
                max_turns=len(self._task_cfg["turns"]),
                task_name=self._task_name,
                task_difficulty=self._task_cfg["difficulty"],
                task_display_name=self._task_cfg["display_name"],
                conversation_history=list(self._history),
                turn_score=0.0,
                hint="",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        response = (action.response or "").strip()

        # Grade the agent's response for the current turn
        turn_cfg = self._task_cfg["turns"][self._current_turn]
        score = _grade_response(
            response=response,
            required_groups=turn_cfg["required_groups"],
            bonus_keywords=turn_cfg.get("bonus", []),
            min_length=turn_cfg.get("min_length", 30),
        )
        self._rewards.append(score)

        # Record in history
        candidate_label = f"Candidate (turn {self._current_turn + 1}): {response}"
        self._history.append(candidate_label)

        hint = _build_hint(score, self._task_name, self._current_turn)

        self._current_turn += 1
        max_turns = len(self._task_cfg["turns"])

        if self._current_turn >= max_turns:
            # Episode complete
            self._done = True
            avg_score = (
                sum(self._rewards) / len(self._rewards) if self._rewards else 0.0
            )
            closing_msg = (
                f"Thank you — that concludes the **{self._task_cfg['display_name']}** "
                f"interview. Your average score was {avg_score:.2f}/1.00. "
                f"Well done!"
            )
            self._history.append(f"Interviewer: {closing_msg}")
            return MultiturnTechnicalInterviewerObservation(
                question=closing_msg,
                turn=self._current_turn,
                max_turns=max_turns,
                task_name=self._task_name,
                task_difficulty=self._task_cfg["difficulty"],
                task_display_name=self._task_cfg["display_name"],
                conversation_history=list(self._history),
                turn_score=score,
                hint=hint,
                done=True,
                reward=round(score, 3),
            )

        # Still in progress — return next question
        next_question = self._task_cfg["turns"][self._current_turn]["question"]
        self._history.append(f"Interviewer: {next_question}")

        return MultiturnTechnicalInterviewerObservation(
            question=next_question,
            turn=self._current_turn,
            max_turns=max_turns,
            task_name=self._task_name,
            task_difficulty=self._task_cfg["difficulty"],
            task_display_name=self._task_cfg["display_name"],
            conversation_history=list(self._history),
            turn_score=score,
            hint=hint,
            done=False,
            reward=round(score, 3),
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Return the current low-level environment state."""
        return self._state
