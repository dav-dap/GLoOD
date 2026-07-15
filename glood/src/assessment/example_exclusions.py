from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TAR_BLOCK_SIZE = 512
EXAMPLES_TO_REMOVE_ARCNAME = "_assessment/examples_to_remove.txt"
EXAMPLES_TO_REMOVE_BASENAME = "examples_to_remove.txt"

ExampleBatch = tuple[int, int, Path]


def _label_prefix(dataset_label: str | None) -> str:
    return f"{dataset_label}: " if dataset_label else ""


@dataclass(frozen=True)
class ExampleExclusionResult:
    kept_batches: list[ExampleBatch]
    excluded_ids: tuple[int, ...]
    skipped_ids: tuple[int, ...]
    missing_ids: tuple[int, ...]
    source_path: Path | None


def _parse_example_token(token: str, *, source_path: Path, line_no: int) -> int:
    token = token.strip()
    if token.startswith("example_"):
        token = token.split("_", 1)[1]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"Invalid example id {token!r} in {source_path}:{line_no}") from exc


def load_example_ids(path: Path) -> tuple[int, ...]:
    ids: set[int] = set()
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            ids.add(_parse_example_token(token, source_path=path, line_no=line_no))
    return tuple(sorted(ids))


def find_examples_to_remove_file(input_root: Path) -> Path | None:
    roots: list[Path] = []
    for root in (input_root, *input_root.parents):
        if root not in roots:
            roots.append(root)

    for root in roots:
        for relative_path in (
            Path(EXAMPLES_TO_REMOVE_ARCNAME),
            Path(EXAMPLES_TO_REMOVE_BASENAME),
        ):
            candidate = root / relative_path
            if candidate.is_file():
                return candidate
    return None


def apply_example_exclusions(
    example_batches: Iterable[ExampleBatch],
    *,
    input_root: Path,
) -> ExampleExclusionResult:
    batches = list(example_batches)
    source_path = find_examples_to_remove_file(input_root)
    if source_path is None:
        return ExampleExclusionResult(
            kept_batches=batches,
            excluded_ids=(),
            skipped_ids=(),
            missing_ids=(),
            source_path=None,
        )

    excluded_ids = set(load_example_ids(source_path))
    present_ids = {int(example_id) for example_id, _, _ in batches}
    skipped_ids = tuple(sorted(present_ids & excluded_ids))
    missing_ids = tuple(sorted(excluded_ids - present_ids))
    kept_batches = [
        (int(example_id), int(epoch), data_dir)
        for example_id, epoch, data_dir in batches
        if int(example_id) not in excluded_ids
    ]

    return ExampleExclusionResult(
        kept_batches=kept_batches,
        excluded_ids=tuple(sorted(excluded_ids)),
        skipped_ids=skipped_ids,
        missing_ids=missing_ids,
        source_path=source_path,
    )


def log_example_exclusions(*, dataset_label: str, result: ExampleExclusionResult) -> None:
    if result.source_path is None:
        checked_names = ", ".join((EXAMPLES_TO_REMOVE_ARCNAME, EXAMPLES_TO_REMOVE_BASENAME))
        print(
            f"{dataset_label}: no exclusion list found after materialization; "
            f"checked {checked_names}; keeping all examples"
        )
        return

    message = (
        f"{dataset_label}: excluded {len(result.skipped_ids)} examples "
        f"listed in {result.source_path}"
    )
    if result.missing_ids:
        preview = ", ".join(str(value) for value in result.missing_ids[:10])
        suffix = "..." if len(result.missing_ids) > 10 else ""
        message = f"{message}; {len(result.missing_ids)} listed ids were not present ({preview}{suffix})"
    print(message)


def _parse_octal_field(field: bytes) -> int:
    stripped = field.split(b"\0", 1)[0].strip()
    if not stripped:
        return 0
    return int(stripped, 8)


def _extract_member_from_tar_tail(archive_path: Path, arcname: str, *, read_bytes: int = 1 << 20) -> bytes | None:
    size = archive_path.stat().st_size
    if size <= 0:
        return None
    read_size = min(size, read_bytes)
    read_size -= read_size % TAR_BLOCK_SIZE
    if read_size <= 0:
        return None

    with archive_path.open("rb") as archive:
        archive.seek(size - read_size)
        tail = archive.read(read_size)

    blocks = [tail[offset : offset + TAR_BLOCK_SIZE] for offset in range(0, len(tail), TAR_BLOCK_SIZE)]
    expected_name = arcname.encode("utf-8")
    for block_idx, block in enumerate(blocks):
        name = block[:100].split(b"\0", 1)[0]
        if name != expected_name:
            continue
        member_size = _parse_octal_field(block[124:136])
        data_start = (block_idx + 1) * TAR_BLOCK_SIZE
        data_end = data_start + member_size
        if data_end > len(tail):
            return None
        return tail[data_start:data_end]
    return None


def _find_materialized_examples_to_remove(staging_dir: Path) -> Path | None:
    candidates = [
        staging_dir / EXAMPLES_TO_REMOVE_ARCNAME,
        staging_dir / EXAMPLES_TO_REMOVE_BASENAME,
    ]
    for child in staging_dir.iterdir():
        if not child.is_dir():
            continue
        candidates.extend(
            [
                child / EXAMPLES_TO_REMOVE_ARCNAME,
                child / EXAMPLES_TO_REMOVE_BASENAME,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _extract_optional_examples_to_remove_with_tar(archive_path: Path, staging_dir: Path) -> Path | None:
    cmd = [
        "tar",
        "-xf",
        str(archive_path),
        "-C",
        str(staging_dir),
        "--wildcards",
        "--no-anchored",
        EXAMPLES_TO_REMOVE_ARCNAME,
        EXAMPLES_TO_REMOVE_BASENAME,
    ]
    # This file is optional; GNU tar returns non-zero when no member matches.
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return _find_materialized_examples_to_remove(staging_dir)


def _write_examples_to_remove(staging_dir: Path, content: bytes) -> Path:
    output_path = staging_dir / EXAMPLES_TO_REMOVE_ARCNAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


def _copy_sidecar_examples_to_remove(
    archive_path: Path,
    staging_dir: Path,
    *,
    dataset_label: str | None,
) -> Path | None:
    support_root = archive_path.parent.parent
    datasets_test_dir = support_root / "datasets_test"
    stem_parts = archive_path.stem.split("-", 1)
    geometry = stem_parts[1] if len(stem_parts) == 2 else archive_path.stem
    sidecar_pattern = f"dataset_{geometry}*_toRemove/{EXAMPLES_TO_REMOVE_BASENAME}"
    if not datasets_test_dir.is_dir():
        print(
            f"{_label_prefix(dataset_label)}sidecar exclusion list not found: "
            f"{datasets_test_dir / sidecar_pattern} (datasets_test directory is missing)"
        )
        return None

    candidates = sorted(datasets_test_dir.glob(sidecar_pattern))
    if not candidates:
        print(
            f"{_label_prefix(dataset_label)}sidecar exclusion list not found: "
            f"{datasets_test_dir / sidecar_pattern}"
        )
        return None
    if len(candidates) > 1:
        names = ", ".join(str(candidate) for candidate in candidates)
        raise ValueError(f"Ambiguous examples-to-remove sidecars for {archive_path}: {names}")

    output_path = _write_examples_to_remove(staging_dir, candidates[0].read_bytes())
    print(
        f"{_label_prefix(dataset_label)}found sidecar exclusion list "
        f"{candidates[0]}; materialized as {output_path}"
    )
    return output_path


def materialize_embedded_examples_to_remove(
    archive_path: Path,
    staging_dir: Path,
    *,
    dataset_label: str | None = None,
) -> Path | None:
    checked_names = ", ".join((EXAMPLES_TO_REMOVE_ARCNAME, EXAMPLES_TO_REMOVE_BASENAME))
    print(f"{_label_prefix(dataset_label)}checking exclusion lists in {archive_path}: {checked_names}")
    if "".join(archive_path.suffixes) == ".tar":
        for arcname in (EXAMPLES_TO_REMOVE_ARCNAME, EXAMPLES_TO_REMOVE_BASENAME):
            content = _extract_member_from_tar_tail(archive_path, arcname)
            if content is not None:
                output_path = _write_examples_to_remove(staging_dir, content)
                print(
                    f"{_label_prefix(dataset_label)}found tar exclusion list "
                    f"{arcname}; materialized as {output_path}"
                )
                return output_path

    extracted_path = _extract_optional_examples_to_remove_with_tar(archive_path, staging_dir)
    if extracted_path is not None:
        print(
            f"{_label_prefix(dataset_label)}found tar exclusion list "
            f"{extracted_path}; using materialized copy"
        )
        return extracted_path

    print(f"{_label_prefix(dataset_label)}tar exclusion lists not found in {archive_path}: {checked_names}")
    return _copy_sidecar_examples_to_remove(archive_path, staging_dir, dataset_label=dataset_label)
