#!/usr/bin/env python3
"""Incremental Python lint gate that preserves a measurable legacy baseline."""
import os
from collections import Counter
from pathlib import Path
import subprocess
import sys


def run(args, source=None):
    return subprocess.run(
        args, input=source, text=True, capture_output=True, check=False
    )


def get_base():
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "pull_request":
        candidate = "origin/" + os.environ.get("GITHUB_BASE_REF", "main")
    elif event == "push":
        candidate = os.environ.get("GITHUB_EVENT_BEFORE", "")
    else:
        candidate = ""

    if candidate and candidate != "0" * 40:
        if run(["git", "rev-parse", "--verify", candidate]).returncode == 0:
            return candidate
    if run(["git", "rev-parse", "--verify", "HEAD^"]).returncode == 0:
        return "HEAD^"
    return None


def source_at(ref, path):
    result = run(["git", "show", f"{ref}:{path}"])
    return result.stdout if result.returncode == 0 else None


def flake8_signatures(path, source):
    result = run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--extend-ignore=E501",
            "--format=%(code)s:%(text)s",
            "--stdin-display-name",
            path,
            "-",
        ],
        source,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or f"flake8 failed for {path}")
    return Counter(line.strip() for line in result.stdout.splitlines() if line.strip())


def black_clean(path, source):
    result = run(
        [sys.executable, "-m", "black", "--check", "--stdin-filename", path, "-"],
        source,
    )
    return result.returncode == 0


def isort_clean(path, source):
    result = run(
        [sys.executable, "-m", "isort", "--check-only", "--filename", path, "-"],
        source,
    )
    return result.returncode == 0


def main():
    base = get_base()
    if base:
        diff = run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{base}...HEAD",
                "--",
                "*.py",
            ]
        )
        files = [line for line in diff.stdout.splitlines() if line]
    else:
        files = []

    if not files:
        print("No changed Python files; incremental lint checks are not applicable.")
        return 0

    failed = False
    for path in files:
        current_path = Path(path)
        if not current_path.is_file():
            continue
        current = current_path.read_text(encoding="utf-8")
        try:
            compile(current, path, "exec")
        except SyntaxError as exc:
            print(f"ERROR: syntax error in {path}:{exc.lineno}: {exc.msg}")
            failed = True
            continue

        baseline = source_at(base, path) if base else None
        if baseline is None:
            print(f"Checking new Python file: {path}")
            commands = [
                [sys.executable, "-m", "black", "--check", path],
                [sys.executable, "-m", "flake8", "--extend-ignore=E501", path],
                [sys.executable, "-m", "isort", "--check-only", path],
            ]
            for command in commands:
                result = run(command)
                if result.returncode:
                    sys.stderr.write(result.stdout + result.stderr)
                    failed = True
            continue

        try:
            compile(baseline, path, "exec")
        except SyntaxError:
            print(
                f"WARNING: {path} had a syntax error in the base revision; "
                "syntax is checked, style checks await a clean baseline."
            )
            continue

        if black_clean(path, baseline):
            result = run([sys.executable, "-m", "black", "--check", path])
            if result.returncode:
                sys.stderr.write(result.stdout + result.stderr)
                failed = True
        else:
            print(f"WARNING: preserving pre-existing Black debt in {path}")

        if isort_clean(path, baseline):
            result = run([sys.executable, "-m", "isort", "--check-only", path])
            if result.returncode:
                sys.stderr.write(result.stdout + result.stderr)
                failed = True
        else:
            print(f"WARNING: preserving pre-existing import-order debt in {path}")

        baseline_lint = flake8_signatures(path, baseline)
        current_lint = flake8_signatures(path, current)
        new_lint = current_lint - baseline_lint
        if new_lint:
            print(f"ERROR: new Flake8 findings in {path}:")
            for finding, count in sorted(new_lint.items()):
                print(f"  {finding} (new occurrences: {count})")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
