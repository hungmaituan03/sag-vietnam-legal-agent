#!/usr/bin/env python3
"""Print a one-line health payload — useful in demos and CI smoke checks."""

from sag_legal.health import healthcheck


def main() -> None:
    print(healthcheck())


if __name__ == "__main__":
    main()
