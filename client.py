# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-turn Technical Interviewer Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        MultiturnTechnicalInterviewerAction,
        MultiturnTechnicalInterviewerObservation,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        MultiturnTechnicalInterviewerAction,
        MultiturnTechnicalInterviewerObservation,
    )


class MultiturnTechnicalInterviewerEnv(
    EnvClient[
        MultiturnTechnicalInterviewerAction,
        MultiturnTechnicalInterviewerObservation,
        State,
    ]
):
    """
    Client for the Multi-turn Technical Interviewer Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling low-latency multi-step interactions.

    Example — connect to a running server::

        with MultiturnTechnicalInterviewerEnv(base_url="http://localhost:8000") as env:
            result = env.reset()
            print(result.observation.question)   # problem statement

            while not result.done:
                result = env.step(
                    MultiturnTechnicalInterviewerAction(response="My answer here...")
                )
                print(result.observation.question)
                print(f"Score: {result.observation.turn_score}")

    Example — start from Docker image::

        env = MultiturnTechnicalInterviewerEnv.from_docker_image("multiturn_technical_interviewer-env:latest")
        try:
            result = env.reset()
            result = env.step(MultiturnTechnicalInterviewerAction(response="..."))
        finally:
            env.close()
    """

    def _step_payload(self, action: MultiturnTechnicalInterviewerAction) -> Dict:
        """Convert action to JSON payload for the step WebSocket message."""
        return {"response": action.response}

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[MultiturnTechnicalInterviewerObservation]:
        """Parse server response into a typed StepResult."""
        obs_data = payload.get("observation", {})

        observation = MultiturnTechnicalInterviewerObservation(
            question=obs_data.get("question", ""),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 5),
            task_name=obs_data.get("task_name", "two_sum"),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            task_display_name=obs_data.get("task_display_name", ""),
            conversation_history=obs_data.get("conversation_history", []),
            turn_score=obs_data.get("turn_score"),
            hint=obs_data.get("hint", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
