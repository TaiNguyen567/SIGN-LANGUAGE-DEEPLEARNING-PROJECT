"""Run local desktop webcam recognition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.realtime import RealtimeRecognizer
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config import feature_config_from_dict, load_config, resolve_project_path
from src.utils.device import get_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()
    try:
        config = load_config(resolve_project_path(args.config, project_root=ROOT))
        device = get_device(args.device)
        checkpoint = resolve_project_path(args.checkpoint, project_root=ROOT)
        feature_config = feature_config_from_dict(config.get("features", {}))
        model, tokenizer, _ = load_model_checkpoint(
            checkpoint,
            device=device,
            expected_feature_dim=feature_config.feature_dim,
            expected_model_config=config.get("model", {}),
        )
        realtime = config.get("realtime", {})
        camera_id = int(args.camera if args.camera is not None else realtime.get("camera_id", 0))
        recognizer = RealtimeRecognizer(
            model, tokenizer, device=device, feature_config=feature_config, config=config,
        )
        for source in recognizer.translator.unavailable_rules(tokenizer.tokens):
            phrase = " ".join(source)
            print(
                f"WARNING: translation rule {phrase!r} cannot be produced by this checkpoint's "
                "vocabulary, so it will never activate. Add matching labeled training data and retrain."
            )
        print("Starting local webcam recognition. Press Q or Esc to exit.")
        recognizer.run(
            camera_id=camera_id,
            width=int(realtime.get("width", 1280)),
            height=int(realtime.get("height", 720)),
            fps=int(realtime.get("fps", 30)),
        )
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Inference unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
