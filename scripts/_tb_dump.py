"""Dump the final scalar values of an rsl_rl training run (temporary helper)."""

import glob
import math
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = sys.argv[1]
files = glob.glob(run_dir + r"\events.out.tfevents.*")
print("tfevents files:", len(files), flush=True)
if not files:
    raise SystemExit(1)

ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]

keys = [
    "Curriculum/lin_vel_cmd_levels",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Metrics/success_rate",
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/base_contact",
    "Episode_Termination/bad_orientation",
]
for k in keys:
    if k not in tags:
        print(f"{k:<46} MISSING", flush=True)
        continue
    ev = ea.Scalars(k)
    print(f"{k:<46} it0={ev[0].value:>9.4f}  last(it{ev[-1].step})={ev[-1].value:>9.4f}", flush=True)

print("--- 辅助惩罚项末值 ---", flush=True)
total = 0.0
for t in sorted(tags):
    if t.startswith("Episode_Reward/"):
        v = ea.Scalars(t)[-1].value
        if v < 0:
            print(f"  {t.split('/', 1)[1]:<24}{v:10.5f}", flush=True)
            total += -v
print(f"  R_aux = {total:.5f}", flush=True)
print(f"  论文 exp(-R_aux)      = {math.exp(-total):.4f}", flush=True)
print(f"  当前 exp(-0.02*R_aux) = {math.exp(-0.02 * total):.4f}", flush=True)
