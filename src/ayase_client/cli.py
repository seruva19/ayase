"""Maintenance commands for the isolated Ayase runtime."""

from __future__ import annotations

import argparse
from . import __version__
from .runtime import RuntimeManager


def main() -> None:
    parser = argparse.ArgumentParser(prog="ayase-client")
    parser.add_argument("command", choices=("install", "doctor"))
    args = parser.parse_args()
    manager = RuntimeManager(__version__)
    if args.command == "install":
        print(manager.install())
        return
    try:
        transport = manager.start()
        health = transport.health()
        print(
            f"Ayase {health['ayase_version']} worker is ready "
            f"(protocol {health['protocol_version']})"
        )
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
