"""python -m app [--config configs/app.yaml] [--host H] [--port P] [--device D]"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Implant site screening app")
    ap.add_argument("--config", default=None, help="defaults to configs/app.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--data-dir", dest="data_dir", default=None)
    args = ap.parse_args()

    import uvicorn

    from app.server import create_app
    from app.settings import load_settings

    settings = load_settings(args.config, host=args.host, port=args.port,
                             device=args.device, data_dir=args.data_dir)
    print(f"Implant site screening -- http://{settings.host}:{settings.port}  "
          f"(device {settings.device}, data in {settings.data_dir})")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port,
                log_level="warning")


if __name__ == "__main__":
    main()
