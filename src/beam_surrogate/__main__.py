"""Dispatch ``python -m beam_surrogate <command>``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: python -m beam_surrogate {generate,train,evaluate} --config configs/cpu.yaml"
        )
        return
    command, rest = args[0], args[1:]
    if command == "generate":
        from beam_surrogate.generate import main as entry
    elif command == "train":
        from beam_surrogate.train import main as entry
    elif command == "evaluate":
        from beam_surrogate.evaluate import main as entry
    else:
        raise SystemExit(f"Unknown command {command!r}. Use generate, train, or evaluate.")
    entry(rest)


if __name__ == "__main__":
    main()
