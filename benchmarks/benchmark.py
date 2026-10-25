from pathlib import Path
import re
import statistics
import subprocess
import sys

executables = sys.argv[1:]

if not executables:
    raise SystemExit(
        "Usage: python benchmark.py <executable>..."
    )

sizes = [256, 512, 1024]
runs = 5

for n in sizes:
    print(f"n = {n}")

    for exe in executables:
        values = []

        for _ in range(runs):
            result = subprocess.run(
                [exe, str(n)],
                capture_output=True,
                text=True,
                check=True,
            )

            match = re.search(
                r"performance = ([0-9.]+) GFLOPS/s",
                result.stdout,
            )

            if match is None:
                raise RuntimeError(
                    f"could not parse output from {exe}"
                )

            values.append(float(match.group(1)))

        print(f"  {Path(exe).name}")
        print(f"    runs   = {values}")
        print(
            f"    median = {statistics.median(values):.3f} GFLOP/s"
        )

    print()
