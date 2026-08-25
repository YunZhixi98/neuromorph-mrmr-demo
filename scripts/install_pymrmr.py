"""Build and install the legacy pymrmr 0.1.11 package.

The PyPI release ships generated C++ that predates Python 3.11, and its Unix
OpenMP flags are not accepted by MSVC. Regenerating the extension with Cython
0.29.37 and adjusting only the Windows compiler flags makes the official
source distribution build in the ``mrmr_demo`` environment.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


VERSION = "0.1.11"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *arguments], check=True)


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Unsafe path in source archive: {member.name}")
    archive.extractall(destination)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pymrmr-build-") as temp_name:
        temp = Path(temp_name)
        run(
            "download",
            f"pymrmr=={VERSION}",
            "--no-deps",
            "--no-binary",
            "pymrmr",
            "--no-build-isolation",
            "--dest",
            str(temp),
        )
        source_archive = next(temp.glob(f"pymrmr-{VERSION}.tar.gz"))
        with tarfile.open(source_archive) as archive:
            safe_extract(archive, temp)

        source = temp / f"pymrmr-{VERSION}"
        setup_path = source / "setup.py"
        setup_text = setup_path.read_text(encoding="utf-8")
        old = "ecompile_args = ['-fopenmp', '-Ofast']\nelink_args = ['-fopenmp']"
        if sys.platform == "win32":
            new = "ecompile_args = ['/O2']\nelink_args = []"
        elif sys.platform == "darwin":
            # Apple Clang does not include OpenMP by default. The teaching
            # subset is small enough that a portable optimized build is ample.
            new = "ecompile_args = ['-O3']\nelink_args = []"
        else:
            new = old

        if new != old:
            if old not in setup_text:
                raise RuntimeError("pymrmr setup.py has changed; compiler patch not applied")
            setup_path.write_text(setup_text.replace(old, new), encoding="utf-8")

        # A newer timestamp forces setup.py to regenerate pymrmr.cpp from the
        # bundled .pyx instead of compiling the Python-3.7-era generated C++.
        (source / "pymrmr.pyx").touch()
        run("install", str(source), "--no-build-isolation")

    import pymrmr  # noqa: F401

    print(f"pymrmr {VERSION} installed and importable")


if __name__ == "__main__":
    main()
