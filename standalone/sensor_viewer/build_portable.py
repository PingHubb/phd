"""Assemble an isolated Windows bundle from official runtime + binary wheels.

Downloads are supplied by the caller; never installs into the host Python or
changes an existing output folder. Only the small launcher is cross-compiled.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import runpy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive entry: {info.filename}")
            source.extract(info, destination)


def build(args):
    source = Path(__file__).resolve().parent
    # Both the repository checkout and the distributed source can be rebuilt.
    app_source = source if (source / "viewer").is_dir() else source / "app"
    helper_source = (source.parents[1] / "phd" / "dependence"
                     if app_source == source else app_source / "phd" / "dependence")
    output = Path(args.output).resolve()
    archive = output.parent / (output.name + ".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("Output directory or ZIP already exists. Choose a new output name.")
    wheels = sorted(Path(args.wheels).glob("*.whl"))
    if len(wheels) != 5 or not all("win_amd64" in p.name or "none-any" in p.name for p in wheels):
        raise ValueError("Expected exactly the five pinned Windows x64/pure-Python wheels.")
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    extract(args.python, runtime)
    site_packages = runtime / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    for wheel in wheels:
        extract(wheel, site_packages)
    pth_files = list(runtime.glob("python*._pth"))
    if len(pth_files) != 1:
        raise ValueError("Expected an official CPython embeddable distribution.")
    pth = pth_files[0]
    pth.write_text(f"{pth.stem}.zip\n.\nLib/site-packages\n../app\nimport site\n", encoding="utf-8")
    app = output / "app"
    app.mkdir()
    shutil.copytree(app_source / "viewer", app / "viewer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(app_source / "run.py", app / "run.py")
    helpers = app / "phd" / "dependence"
    helpers.mkdir(parents=True)
    for name in ("__init__.py", "sensor_api.py", "sensor_protocol.py"):
        shutil.copy2(helper_source / name, helpers / name)
    shutil.copy2(helper_source.parent / "__init__.py", app / "phd" / "__init__.py")
    for name in ("README.md", "Diagnose.cmd", "requirements-windows.txt", "launcher.c", "build_portable.py"):
        shutil.copy2(source / name, output / name)
    tests = source / ("tests" if (source / "tests").is_dir() else "source-tests")
    shutil.copytree(tests, output / "source-tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    exe = output / "PingLabSensorViewer.exe"
    subprocess.run([str(Path(args.zig).resolve()), "cc", "-target", "x86_64-windows-gnu",
                    "-O2", "-municode", "-Wl,--subsystem,windows", str(source / "launcher.c"),
                    "-o", str(exe), "-luser32"], check=True)
    if exe.read_bytes()[:2] != b"MZ":
        raise ValueError("Launcher is not a Windows PE executable.")
    dependencies = dict(python=dict(file=Path(args.python).name, sha256=sha256(args.python)),
                        wheels=[dict(file=p.name, sha256=sha256(p)) for p in wheels])
    dependencies.update(
        app_version=runpy.run_path(str(app_source / "viewer" / "__init__.py"))["VERSION"],
        target="Windows x64", python_version="3.13.17",
        launcher_compiler=subprocess.check_output([args.zig, "version"], text=True).strip(),
        physical_windows_test="0.1.0 launcher confirmed working by user; this updated 0.2.0 plotter requires a Windows test",
        source_helpers={p.name: sha256(p) for p in helpers.glob("*.py")},
        official_sources=["https://www.python.org/downloads/windows/", "https://pypi.org/", "https://ziglang.org/download/"],
    )
    (output / "BUILD_INFO.json").write_text(json.dumps(dependencies, indent=2)+"\n", encoding="utf-8")
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS.txt").write_text("\n".join(manifest)+"\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(output.name) / path.relative_to(output))
    print(json.dumps(dict(zip=str(archive), sha256=sha256(archive), bytes=archive.stat().st_size,
                          executable=str(exe)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, help="Official Windows embeddable ZIP")
    parser.add_argument("--wheels", required=True)
    parser.add_argument("--zig", required=True)
    parser.add_argument("--output", required=True, help="New folder; matching ZIP must not exist")
    build(parser.parse_args())
