"""Temporary probe: import health + installed versions in the isaac env."""

import importlib.metadata as md

MODULES = [
    "typing_extensions", "psutil", "click", "websockets",
    "onnx", "rerun", "viser", "wandb",
    "torch", "gymnasium", "isaaclab", "rl_games",
]

for m in MODULES:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "?")
        print(f"{m:<18} import OK    version={ver}")
    except Exception as exc:  # noqa: BLE001
        print(f"{m:<18} import FAIL  {type(exc).__name__}: {exc}")

print("\ninstalled distribution versions:")
for name in [
    "typing-extensions", "psutil", "click", "websockets",
    "onnx", "rerun-sdk", "viser", "wandb", "isaacsim-kernel", "rl-games",
]:
    try:
        print(f"  {name:<20} {md.version(name)}")
    except Exception:  # noqa: BLE001
        print(f"  {name:<20} <not installed>")
