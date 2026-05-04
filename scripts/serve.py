"""Launch the TTS serving API."""
from __future__ import annotations

import argparse

import uvicorn

from tts_pipeline.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS Serving API")
    parser.add_argument("--bundle-path", default="artifacts/export")
    parser.add_argument("--shadow-bundle-path", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app = create_app(args.bundle_path, args.shadow_bundle_path)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, reload=args.reload)


if __name__ == "__main__":
    main()
