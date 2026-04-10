# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Multi-turn Technical Interviewer Environment.

The environment simulates a technical interview: the agent receives a problem,
submits a solution, then navigates follow-up questions covering time/space
complexity, edge cases, concurrency, and distributed systems design.

Three tasks of increasing difficulty:
  - two_sum       (easy)   : algorithmic problem with 5 follow-up turns
  - lru_cache     (medium) : data-structure design with 6 follow-up turns
  - rate_limiter  (hard)   : distributed systems design with 7 follow-up turns
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MultiturnTechnicalInterviewerAction(Action):
    """
    Agent action: the agent's textual response to the interviewer's question.

    This is the agent's spoken answer during the interview — a solution,
    explanation, code snippet, or design discussion as appropriate.
    """

    response: str = Field(
        ...,
        description=(
            "The agent's complete answer to the interviewer's current question. "
            "Should include reasoning, code (if applicable), complexity analysis, "
            "and discussion of trade-offs."
        ),
    )


class MultiturnTechnicalInterviewerObservation(Observation):
    """
    Observation returned by the environment after each step.

    Contains the interviewer's current question, contextual metadata, and a
    running history of the conversation for the agent to refer back to.
    """

    question: str = Field(
        default="",
        description=(
            "The interviewer's current question or prompt. On reset() this is "
            "the full problem statement. On each step() it is the follow-up."
        ),
    )
    turn: int = Field(
        default=0,
        description=(
            "Current turn index. 0 = problem statement (from reset). "
            "1..max_turns = graded follow-up turns."
        ),
    )
    max_turns: int = Field(
        default=5,
        description="Total number of graded turns (steps) in this task episode.",
    )
    task_name: str = Field(
        default="two_sum",
        description="Identifier for the current task: two_sum | lru_cache | rate_limiter.",
    )
    task_difficulty: str = Field(
        default="easy",
        description="Difficulty label for this task: easy | medium | hard.",
    )
    task_display_name: str = Field(
        default="Two Sum",
        description="Human-readable name of the current task.",
    )
    conversation_history: List[str] = Field(
        default_factory=list,
        description=(
            "Chronological list of conversation entries. Entries alternate between "
            "'Interviewer: <question>' and 'Candidate: <response>' strings."
        ),
    )
    turn_score: Optional[float] = Field(
        default=None,
        description=(
            "Score [0.0, 1.0] awarded for the most recent response, or None on reset. "
            "Reflects correctness, use of relevant terminology, and depth."
        ),
    )
    hint: str = Field(
        default="",
        description=(
            "Optional hint or feedback from the interviewer about the previous answer. "
            "Empty string when no hint is provided."
        ),
    )
