"""
Multi-turn Technical Interviewer — Inference Script
====================================================

MANDATORY ENVIRONMENT VARIABLES
---------------------------------
  API_BASE_URL    LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME      Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN        Hugging Face / API key
  IMAGE_NAME      Docker image name (optional; if set, spins up a container)
  BASE_URL        Direct server URL (used when IMAGE_NAME is not set;
                  default: http://localhost:8000)

STDOUT FORMAT (strictly required by the evaluator)
---------------------------------------------------
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Five episodes are run in sequence, one per task (server cycles TASK_ORDER):
  1. two_sum        (easy)
  2. lru_cache      (medium)
  3. median_stream  (medium)
  4. rate_limiter   (hard)
  5. message_queue  (hard)

  Override count with NUM_EPISODES (e.g. NUM_EPISODES=1 for a single episode).

  OUTPUT_DIR       Directory for baseline_scores.json (default: outputs)
"""

import asyncio
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from baseline_scores_output import write_baseline_scores
from models import MultiturnTechnicalInterviewerAction
from client import MultiturnTechnicalInterviewerEnv

# Ensure TLS verification works on macOS Python builds that miss system certs.
# This is especially relevant for wss://*.hf.space connections.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME")
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")

API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK: str = "multiturn_technical_interviewer"

# One full pass over server TASK_ORDER (two_sum → … → message_queue)
NUM_EPISODES: int = max(1, int(os.getenv("NUM_EPISODES", "5")))

# Per-episode limits
MAX_STEPS: int = 10          # safety ceiling; episodes end via done=True
SUCCESS_SCORE_THRESHOLD: float = 0.40   # average reward >= this → success
TEMPERATURE: float = 0.3     # lower temperature for more focused answers
MAX_TOKENS: int = 280        # short answers reduce LLM latency → fewer WS keepalive timeouts

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    # Collapse newlines so the entire action fits on one line
    action_oneline = action.replace("\n", " ").replace("\r", " ").strip()
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_oneline!r} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt — instructs the model to behave like a strong candidate
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an experienced software engineer being interviewed at a top-tier
    technology company for a senior engineering position.

    Your goal is to give clear, technically precise, and well-reasoned answers
    to every question the interviewer asks.

    Guidelines:
    - Always state the time complexity (Big-O notation) and space complexity
      of your solutions.
    - For algorithms and data-structure questions, provide working pseudocode
      or actual code snippets.
    - For system design questions, name concrete data structures, algorithms,
      and real tools (e.g. Redis, consistent hashing, doubly-linked list).
    - Discuss trade-offs explicitly: when you prefer one approach over another
      and why.
    - Cover edge cases proactively.
    - For distributed systems questions, address: sharding, replication,
      consistency, failure handling, and latency.
    - Keep each answer brief: at most ~180 words, tight bullets or short paragraphs.
      Long answers risk timing out the connection to the environment server.
    - Reply with ONLY your answer — no meta-commentary like
      "As a candidate, I would say...".
    """
).strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    task_name: str,
    task_difficulty: str,
    task_display_name: str,
    turn: int,
    max_turns: int,
    question: str,
    history: List[str],
    last_hint: str,
    last_score: Optional[float],
) -> str:
    history_block = (
        "\n".join(history[-8:]) if history else "No prior exchanges."
    )
    hint_block = (
        f"\nInterviewer hint on your last answer: {last_hint}"
        if last_hint
        else ""
    )
    score_block = (
        f"\n(Your last turn score: {last_score:.2f}/1.00)"
        if last_score is not None
        else ""
    )

    return textwrap.dedent(
        f"""
        === TECHNICAL INTERVIEW ===
        Task      : {task_display_name} [{task_difficulty}]
        Turn      : {turn}/{max_turns}
        {score_block}{hint_block}

        --- Conversation so far ---
        {history_block}
        ---------------------------

        Interviewer's current question:
        {question}

        Your answer:
        """
    ).strip()


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def get_model_response(
    client: OpenAI,
    task_name: str,
    task_difficulty: str,
    task_display_name: str,
    turn: int,
    max_turns: int,
    question: str,
    history: List[str],
    last_hint: str,
    last_score: Optional[float],
) -> str:
    user_prompt = build_user_prompt(
        task_name=task_name,
        task_difficulty=task_difficulty,
        task_display_name=task_display_name,
        turn=turn,
        max_turns=max_turns,
        question=question,
        history=history,
        last_hint=last_hint,
        last_score=last_score,
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "I need a moment to think about this."
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "I need a moment to think about this."


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    env: MultiturnTechnicalInterviewerEnv,
    client: OpenAI,
) -> Dict[str, Any]:
    """Run one full interview episode and emit START / STEP* / END logs."""
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False
    task_name: str = "unknown"
    task_difficulty: Optional[str] = None

    # Reset — get problem statement
    try:
        reset_result = await env.reset()
    except Exception as exc:
        print(f"[DEBUG] env.reset() failed: {exc}", flush=True)
        log_start(task="error", env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        try:
            await env.disconnect()
        except Exception:
            pass
        return {
            "task_name": "error",
            "task_difficulty": None,
            "steps": 0,
            "episode_score": 0.0,
            "success": False,
            "rewards": [],
            "error": str(exc),
        }

    obs = reset_result.observation
    task_name = obs.task_name
    task_difficulty = obs.task_difficulty

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    current_question = obs.question
    history: List[str] = list(obs.conversation_history)
    last_hint: str = ""
    last_score: Optional[float] = None
    result = reset_result

    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Run the sync OpenAI call in a thread so the asyncio loop can still
            # service WebSocket pings; blocking here caused keepalive timeouts (1011).
            response = await asyncio.to_thread(
                lambda: get_model_response(
                    client=client,
                    task_name=obs.task_name,
                    task_difficulty=obs.task_difficulty,
                    task_display_name=obs.task_display_name,
                    turn=obs.turn,
                    max_turns=obs.max_turns,
                    question=current_question,
                    history=history,
                    last_hint=last_hint,
                    last_score=last_score,
                )
            )

            # Step the environment
            try:
                result = await env.step(
                    MultiturnTechnicalInterviewerAction(response=response)
                )
            except Exception as exc:
                err_msg = str(exc)
                print(f"[DEBUG] env.step() failed at step {step}: {err_msg}", flush=True)
                log_step(step=step, action=response, reward=0.0, done=True, error=err_msg)
                steps_taken = step
                break

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            current_question = obs.question
            history = list(obs.conversation_history)
            last_hint = obs.hint or ""
            last_score = obs.turn_score

            log_step(step=step, action=response, reward=reward, done=done, error=error)

            if done:
                break

        # Compute normalised episode score
        if rewards:
            score = sum(rewards) / len(rewards)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        # Drop the WebSocket so the next episode opens a fresh connection.
        try:
            await env.disconnect()
        except Exception:
            pass

    return {
        "task_name": task_name,
        "task_difficulty": task_difficulty,
        "steps": steps_taken,
        "episode_score": round(score, 6),
        "success": success,
        "rewards": [round(r, 4) for r in rewards],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    episodes: List[Dict[str, Any]] = []

    if IMAGE_NAME:
        # Docker mode: start ONE container for all episodes.
        # The server's module-level counter cycles tasks on each reset().
        # Using a single container avoids creating fresh containers (each
        # with counter=0) which would always return the first task.
        env = await MultiturnTechnicalInterviewerEnv.from_docker_image(IMAGE_NAME)
        try:
            for _ in range(NUM_EPISODES):
                episodes.append(await run_episode(env, client))
        finally:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)
    else:
        # Server mode (BASE_URL): fresh WS connection per episode to avoid
        # keepalive timeout during slow LLM calls.  The server's module-level
        # counter persists across connections within the same process, so task
        # cycling works correctly.
        for _ in range(NUM_EPISODES):
            env = MultiturnTechnicalInterviewerEnv(base_url=BASE_URL)
            try:
                episodes.append(await run_episode(env, client))
            finally:
                try:
                    await env.close()
                except Exception as exc:
                    print(f"[DEBUG] env.close() error: {exc}", flush=True)

    write_baseline_scores(
        episodes,
        script=Path(__file__).name,
        benchmark=BENCHMARK,
        model=MODEL_NAME,
        extras={
            "api_base_url": API_BASE_URL,
            "num_episodes_requested": NUM_EPISODES,
            "base_url": BASE_URL,
            "image_name": IMAGE_NAME,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
