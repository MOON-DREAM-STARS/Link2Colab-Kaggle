"""Tiny stand-in training script: print progress and write outputs/result.txt."""

from pathlib import Path


def main() -> None:
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    result = output / "result.txt"
    total = 3
    for step in range(1, total + 1):
        print(f"step {step}/{total}", flush=True)
    result.write_text("status=ok\n", encoding="utf-8")
    print(f"wrote {result.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
