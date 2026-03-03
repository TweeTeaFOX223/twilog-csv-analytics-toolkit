from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from typing import Iterable

LICENSE_TEXT_PATH = Path("app/static/THIRD_PARTY_LICENSES.txt")
LICENSE_JSON_PATH = Path("app/static/THIRD_PARTY_LICENSES.json")
PROJECT_NAME = "twilog-analytics"
EXCLUDED_PACKAGE_NAMES = {
    "altgraph",
    "httpx",
    "iniconfig",
    "pluggy",
    "pygments",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pytest",
    "pytest-asyncio",
}


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp932", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _normalize_license(meta: metadata.PackageMetadata) -> str:
    license_expression = (meta.get("License-Expression") or "").strip()
    if license_expression:
        return license_expression

    license_field = (meta.get("License") or "").strip()
    if (
        license_field
        and license_field.upper() not in {"UNKNOWN", "NONE"}
        and "\n" not in license_field
        and len(license_field) <= 120
    ):
        return license_field

    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    if classifiers:
        return classifiers[-1].split("::")[-1].strip()

    return "UNKNOWN"


def _license_files(dist: metadata.Distribution) -> list[str]:
    files = [f.strip() for f in (dist.metadata.get_all("License-File") or []) if f.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in files:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _read_license_file(dist: metadata.Distribution, relative_path: str) -> str:
    target = dist.locate_file(relative_path)
    if not target.is_file():
        return ""
    return _decode_bytes(target.read_bytes()).strip()


def build_records(include_names: set[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name:
            continue
        if name.lower() == PROJECT_NAME:
            continue
        if name.lower() in EXCLUDED_PACKAGE_NAMES:
            continue
        if include_names is not None and name.lower() not in include_names:
            continue

        record = {
            "name": name,
            "version": dist.version,
            "license": _normalize_license(dist.metadata),
            "summary": (dist.metadata.get("Summary") or "").strip(),
            "home_page": (dist.metadata.get("Home-page") or "").strip(),
            "license_files": [],
        }

        for license_rel in _license_files(dist):
            license_text = _read_license_file(dist, license_rel)
            if not license_text:
                continue
            record["license_files"].append({"path": license_rel, "text": license_text})

        rows.append(record)

    rows.sort(key=lambda x: x["name"].lower())
    return rows


def _render_txt(records: Iterable[dict]) -> str:
    lines: list[str] = []
    lines.append("Third-Party Licenses")
    lines.append("====================")
    lines.append("")
    lines.append(
        "This file lists third-party Python packages used by this application and bundled at runtime."
    )
    lines.append("")

    for rec in records:
        lines.append(f"Package: {rec['name']}")
        lines.append(f"Version: {rec['version']}")
        lines.append(f"License: {rec['license']}")
        if rec["summary"]:
            lines.append(f"Summary: {rec['summary']}")
        if rec["home_page"]:
            lines.append(f"Home-page: {rec['home_page']}")
        if rec["license_files"]:
            lines.append("License Files:")
            for lf in rec["license_files"]:
                lines.append(f"--- BEGIN {lf['path']} ---")
                lines.append(lf["text"])
                lines.append(f"--- END {lf['path']} ---")
        else:
            lines.append("License Files: (not declared in package metadata)")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    return "\n".join(lines)


def _scan_internal_package_names(internal_dir: Path) -> set[str]:
    names: set[str] = set()
    if not internal_dir.exists():
        return names

    # Fast path: dist-info directories that survived collection.
    for item in internal_dir.iterdir():
        if item.is_dir() and item.name.endswith(".dist-info"):
            names.add(item.name.split("-")[0].lower())

    # Fallback: infer from top-level files/directories found under _internal.
    present_entries = {p.name.lower() for p in internal_dir.iterdir()}
    for dist in metadata.distributions():
        dist_name = (dist.metadata.get("Name") or "").strip()
        if not dist_name:
            continue
        if dist_name.lower() == PROJECT_NAME:
            continue
        for file_item in dist.files or []:
            top = str(file_item).split("/", 1)[0]
            if not top or top.endswith(".dist-info") or top.endswith(".data"):
                continue
            if top.lower() in present_entries:
                names.add(dist_name.lower())
                break

    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bundled third-party license report")
    parser.add_argument("--txt", type=Path, default=LICENSE_TEXT_PATH)
    parser.add_argument("--json", type=Path, default=LICENSE_JSON_PATH)
    parser.add_argument(
        "--internal-dir",
        type=Path,
        default=None,
        help="Optional: dist internal folder (e.g. dist/twilog-analytics/_internal) to filter packages",
    )
    args = parser.parse_args()

    include_names = (
        _scan_internal_package_names(args.internal_dir) if args.internal_dir else None
    )
    records = build_records(include_names=include_names)

    args.txt.parent.mkdir(parents=True, exist_ok=True)
    args.txt.write_text(_render_txt(records), encoding="utf-8")
    args.json.write_text(
        json.dumps({"packages": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(records)} package records to {args.txt} and {args.json}")


if __name__ == "__main__":
    main()
