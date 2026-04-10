---
title: Multi-turn Technical Interviewer
emoji: 🎤
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - technical-interview
  - evaluation
  - algorithms
  - system-design
---

# Multi-turn Technical Interviewer

An OpenEnv environment that simulates a realistic technical interview.  
The agent receives a coding or system-design problem, submits a solution,
then navigates **5–7 structured follow-up questions** covering time/space
complexity, edge cases, concurrency, and distributed systems design.

Each answer is graded with a deterministic keyword-matching rubric, providing
**dense per-turn reward signals** throughout the episode. A bad early answer
compounds — the environment is designed to stress multi-turn reasoning under
natural difficulty escalation.

---

## Environment Overview

| Property | Value |
|---|---|
| **Action space** | Free-form text (`response: str`) |
| **Observation** | Interviewer question + full conversation history |
| **Reward** | `[0.0, 1.0]` per turn via keyword grading |
| **Episode length** | 5–7 turns depending on task |
| **Tasks** | 3 (easy → medium → hard) |

---

## Action Space

```python
class MultiturnTechnicalInterviewerAction(Action):
    response: str   # The agent's complete answer to the current question
```

The agent's response should be a free-form text string containing reasoning,
code, Big-O analysis, trade-off discussion, etc. — whatever best answers the
interviewer's current question.

---

## Observation Space

```python
class MultiturnTechnicalInterviewerObservation(Observation):
    question: str                    # Interviewer's current question
    turn: int                        # Current turn (0 = problem statement)
    max_turns: int                   # Total graded turns in this episode
    task_name: str                   # "two_sum" | "lru_cache" | "rate_limiter"
    task_difficulty: str             # "easy" | "medium" | "hard"
    task_display_name: str           # Human-readable task name
    conversation_history: List[str]  # Full conversation so far
    turn_score: Optional[float]      # Score for the most recent answer [0,1]
    hint: str                        # Interviewer feedback on last answer
    done: bool                       # True when episode is complete
    reward: Optional[float]          # Reward for the most recent step
```

---

## Tasks

### Task 1 — Two Sum (Easy, 5 turns)

**Problem:** Given `nums` and `target`, return indices of two numbers that sum
to target. Solve and discuss the design.

| Turn | Question |
|------|----------|
| 0 | Problem statement (presented on `reset()`) |
| 1 | Time complexity — can you beat O(n²) with a hash map? |
| 2 | Space complexity trade-off — O(1) vs O(n) space |
| 3 | Edge cases — empty array, no solution, duplicates, negatives |
| 4 | Distributed approach — 10B elements across 100 machines |
| 5 | Final trade-off summary — single-node vs distributed |

**Grading keywords:** hash map, O(n), complement, space trade-off, edge cases,
partitioned, distributed, coordinator, network overhead.

---

### Task 2 — LRU Cache (Medium, 6 turns)

**Problem:** Design a Least Recently Used cache with O(1) `get` and `put`.

| Turn | Question |
|------|----------|
| 0 | Problem statement |
| 1 | Why is O(1) achieved? Role of each data structure |
| 2 | Edge cases — capacity 1, update existing key, cache miss |
| 3 | Thread safety — concurrent `get`/`put`, race conditions |
| 4 | Distributed design — partition across servers |
| 5 | Cache invalidation — diverged replicas, consistency |
| 6 | Final review — complexity and bottleneck |

**Grading keywords:** doubly linked list, hash map, O(1), mutex/lock,
consistent hashing, Redis, write-through, TTL, eventual consistency.

---

### Task 3 — Distributed Rate Limiter (Hard, 7 turns)

**Problem:** Design a rate limiter enforcing 100 req/user/min at scale (50+ servers).

| Turn | Question |
|------|----------|
| 0 | Problem statement |
| 1 | Algorithm choice — token bucket vs sliding window trade-offs |
| 2 | Sliding window implementation & complexity |
| 3 | Race condition fix — atomic operations |
| 4 | Global rate limit across 50 servers |
| 5 | Redis outage — graceful degradation design |
| 6 | Anti-evasion — user splits traffic across 50 IPs |
| 7 | Final review — full design summary |

**Grading keywords:** token bucket, sliding window, Redis, Lua script,
atomic, centralized store, fail open, circuit breaker, OAuth/API key.

---

## Reward Function

Each turn produces a reward in **[0.0, 1.0]** computed as:

```
score = required_matched/total_required * 0.75
      + bonus_matched/total_bonus       * 0.25
      * length_factor                         (penalty for <30-char answers)
      + 0.04 if len(response) > 300 chars
```

**Score bands:**
| Score | Meaning |
|-------|---------|
| 0.0 | Empty / dismissive response |
| 0.1–0.3 | Thin or off-topic |
| 0.3–0.6 | Partial credit — some relevant concepts |
| 0.6–0.8 | Solid answer, most key terms covered |
| 0.8–1.0 | Excellent — required + bonus topics addressed |

**Episode score** = mean reward across all turns.  
**Success** = episode score ≥ 0.40.

---

## Episode Flow

```
reset()              → Observation(question=<problem>, turn=0, done=False, reward=0.0)
step(response_1)     → Observation(question=<follow_up_1>, turn=1, reward=r1)
step(response_2)     → Observation(question=<follow_up_2>, turn=2, reward=r2)
…
step(response_N)     → Observation(question=<closing>, turn=N, done=True, reward=rN)
```

The environment **auto-cycles through tasks** on consecutive `reset()` calls:
`two_sum → lru_cache → rate_limiter → two_sum → …`

Set `INTERVIEW_TASK=lru_cache` (env var) to pin the starting task.

---

## Quick Start

### Running locally

```bash
# Install dependencies
pip install openenv-core uvicorn fastapi

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Running with Docker

```bash
# Build the image
docker build -t multiturn_technical_interviewer-env:latest .

# Run the server
docker run -p 8000:8000 multiturn_technical_interviewer-env:latest
```

### Running inference

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

# Against a running server:
export BASE_URL=http://localhost:8000
python inference.py

# Against a Docker image:
export IMAGE_NAME=multiturn_technical_interviewer-env:latest
python inference.py
```

Expected output (three tasks, one block each):

```
[START] task=two_sum env=multiturn_technical_interviewer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='...' reward=0.72 done=false error=null
[STEP] step=2 action='...' reward=0.68 done=false error=null
...
[END] success=true steps=5 score=0.693 rewards=0.72,0.68,0.71,0.65,0.66

[START] task=lru_cache env=multiturn_technical_interviewer model=Qwen/Qwen2.5-72B-Instruct
...
```

---

## Project Structure

```
multiturn_technical_interviewer/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest with task definitions
├── pyproject.toml         # Project metadata and dependencies
├── Dockerfile             # Container image definition
├── validate-submission.sh # Pre-submission validation script
├── client.py              # MultiturnTechnicalInterviewerEnv HTTP/WS client
├── models.py              # Action and Observation Pydantic models
├── inference.py           # Baseline inference script (all 3 tasks)
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI application
    └── multiturn_technical_interviewer_environment.py  # Core environment logic
```

---

## Baseline Scores

Baseline run with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Turns | Baseline Score |
|------|-----------|-------|----------------|
| two_sum | Easy | 5 | ~0.68 |
| lru_cache | Medium | 6 | ~0.63 |
| rate_limiter | Hard | 7 | ~0.58 |

Scores reflect the keyword-graded rubric. Frontier models (GPT-4o, Claude 3.5)
typically achieve 0.75–0.90. The hard task (rate_limiter) genuinely challenges
models with its multi-layered systems design requirements.

---

## OpenEnv API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode; returns problem statement |
| `POST` | `/step` | Submit `{"response": "..."}`, get next question + reward |
| `GET` | `/state` | Current episode state (turn count, episode_id) |
| `GET` | `/schema` | Action / Observation JSON schemas |
| `WS` | `/ws` | Persistent WebSocket session |
| `GET` | `/health` | Container health check |
| `GET` | `/web` | Interactive web UI |
