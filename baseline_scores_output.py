# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write ``outputs/baseline_scores.json`` after inference runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASELINE_FILENAME = "baseline_scores.json"


def default_output_dir() -> str:
    return os.environ.get("OUTPUT_DIR", "outputs")


def write_baseline_scores(
    episodes: List[Dict[str, Any]],
    *,
    script: str,
    benchmark: str,
    model: str,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write baseline episode metrics to ``{OUTPUT_DIR}/baseline_scores.json``.

    Returns the absolute path written.
    """
    out_dir = default_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, BASELINE_FILENAME)

    scored = [e["episode_score"] for e in episodes if "episode_score" in e]
    mean_episode = sum(scored) / len(scored) if scored else 0.0

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "benchmark": benchmark,
        "model": model,
        "output_dir": out_dir,
        "mean_episode_score": round(mean_episode, 6),
        "num_episodes": len(episodes),
        "episodes": episodes,
    }
    if extras:
        payload["config"] = extras

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"[INFO] Wrote baseline scores to {path}", flush=True)
    return os.path.abspath(path)
