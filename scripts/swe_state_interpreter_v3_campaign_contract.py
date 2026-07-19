#!/usr/bin/env python3
"""Fail-closed campaign preflight for the exact V3 SWE development campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V3_RUNS_ROOT = ROOT / "runs/swe_state_interpreter_v3_development"
CHECKER_PATH = ROOT / "scripts/check_swe_task_state_v3_development_cohort.py"
# Updated only after the declaration/checker/protocol byte chain is finalized.
CHECKER_SHA256 = "0b0ddc053669fab6ef6c37ddd26ee523d66a135d7515bc9c6dece10ff979a21c"
CAMPAIGN_KIND = "swe_verified_behavioral_trajectory_campaign"
IMAGE_MANIFEST_KIND = "swe_verified_behavioral_campaign_image_manifest"
CAMPAIGN_TASK_COUNT = 30
EXPECTED_CAMPAIGNS = {
    "configs/swe_task_state_v3_development_a_campaign.json": (
        "4379a32a60d3772421239c5bd2c27fa07e56089810a0adcd71bfb951caa0a0a2"
    ),
    "configs/swe_task_state_v3_development_b_campaign.json": (
        "98dab25c7f11e46fe7b29f72a15535d11c276f4d68b9b68c01ba7f1bfad53387"
    ),
}
EXPECTED_RUN_LABELS = {
    "configs/swe_task_state_v3_development_a_campaign.json": (
        "swe_task_state_v3_development_a_20260719"
    ),
    "configs/swe_task_state_v3_development_b_campaign.json": (
        "swe_task_state_v3_development_b_20260719"
    ),
}
EXPECTED_SELECTION_PROOF_PATH = (
    "validation/swe-task-state-v3-development-cohort-selection.json"
)
EXPECTED_SELECTION_PROOF_SHA256 = (
    "7adb31c20ae3b0fe8e0074e921afc0847f11e42d48e36863741cc09f4a86b9bf"
)
EXPECTED_DATASET = {
    "repo_id": "princeton-nlp/SWE-bench_Verified",
    "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
}
EXPECTED_GENERATION = {
    "model_repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "served_model": "qwen3.6-27b-nvfp4",
    "qwen_code_version": "0.19.4",
    "max_model_len": 65536,
    "max_session_turns": 50,
    "agent_wall_seconds": 900,
    "retain_empty_predictions": True,
}
INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REFERENCE_RE = re.compile(
    r"^swebench/sweb\.eval\.x86_64\.[A-Za-z0-9_.-]+@sha256:[0-9a-f]{64}$"
)
RUN_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def strict_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    payload = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    return dict(mapping(value, label)), payload


def _lexical_absolute(path: Path) -> Path:
    candidate = path.expanduser()
    require(".." not in candidate.parts, f"non-canonical path: {path}")
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return Path(os.path.abspath(candidate))


def _exact_repository_file(path: Path, expected: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    expected_absolute = _lexical_absolute(expected)
    require(candidate == expected_absolute, f"{label} is not the exact declared path")
    require(candidate.is_file() and not candidate.is_symlink(), f"{label} is not a regular file")
    require(candidate.resolve(strict=True) == candidate, f"{label} traverses a symlink")
    require(candidate.is_relative_to(ROOT), f"{label} must be inside the repository")
    return candidate


def _load_pinned_checker() -> types.ModuleType:
    checker_path = _exact_repository_file(CHECKER_PATH, CHECKER_PATH, "V3 declaration checker")
    payload = checker_path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    require(observed == CHECKER_SHA256, "V3 declaration checker SHA-256 changed")
    module_name = f"_pinned_swe_state_v3_campaign_checker_{observed}"
    module = types.ModuleType(module_name)
    module.__file__ = str(checker_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(payload, str(checker_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    require(callable(getattr(module, "validate_declaration", None)), "V3 checker API changed")
    return module


def docker_tag(instance_id: str) -> str:
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_") + ":latest"


def inspect_local_image(tag: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(
            subprocess.check_output(["docker", "image", "inspect", tag], text=True)
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"required cached image is missing: {tag}") from error
    require(isinstance(value, list), f"unexpected image inspection result: {tag}")
    return value


def validate_campaign_contract(
    campaign: Mapping[str, Any],
    image_registry: Mapping[str, Any],
    *,
    image_inspector: Callable[[str], Sequence[Mapping[str, Any]]] = inspect_local_image,
    forbidden_instance_ids: frozenset[str] = frozenset(),
) -> tuple[list[str], list[dict[str, Any]]]:
    """Validate frozen generation pins and exact local image-byte coverage."""

    require(campaign.get("schema_version") == 1, "campaign schema mismatch")
    require(campaign.get("kind") == CAMPAIGN_KIND, "campaign kind mismatch")
    require(campaign.get("dataset") == EXPECTED_DATASET, "campaign dataset pin mismatch")
    selection = mapping(campaign.get("selection"), "campaign selection")
    require(selection.get("lens_outputs_used") is False, "campaign selection used lens output")
    require(selection.get("official_outcomes_used") is False, "campaign selection used official outcomes")
    require(campaign.get("generation") == EXPECTED_GENERATION, "campaign frozen generation settings mismatch")
    raw_instance_ids = campaign.get("instance_ids")
    require(isinstance(raw_instance_ids, list), "campaign instance_ids must be an array")
    instance_ids = list(raw_instance_ids)
    require(len(instance_ids) == CAMPAIGN_TASK_COUNT, "V3 campaign must contain exactly 30 tasks")
    require(
        all(isinstance(value, str) and INSTANCE_ID_RE.fullmatch(value) is not None for value in instance_ids),
        "campaign contains an invalid SWE instance ID",
    )
    require(len(instance_ids) == len(set(instance_ids)), "campaign instance IDs repeat")
    overlap = sorted(set(instance_ids) & forbidden_instance_ids)
    require(not overlap, f"campaign overlaps protected prior/reserved tasks: {overlap}")

    require(image_registry.get("schema_version") == 1, "image registry schema mismatch")
    registry_images = mapping(image_registry.get("images"), "image registry images")
    require(set(registry_images) == set(instance_ids), "image registry coverage must exactly equal campaign instance coverage")
    inspected_images: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        tag = docker_tag(instance_id)
        architecture_pin = mapping(
            mapping(registry_images[instance_id], f"image pin {instance_id}").get("x86_64"),
            f"x86_64 image pin {instance_id}",
        )
        require(set(architecture_pin) == {"reference", "image_id"}, f"image pin fields changed: {instance_id}")
        reference = architecture_pin.get("reference")
        image_id = architecture_pin.get("image_id")
        require(isinstance(reference, str) and REFERENCE_RE.fullmatch(reference) is not None, f"invalid RepoDigest pin: {instance_id}")
        require(reference.startswith(f"{tag.removesuffix(':latest')}@sha256:"), f"RepoDigest pin does not name the campaign image: {instance_id}")
        require(isinstance(image_id, str) and SHA256_RE.fullmatch(image_id) is not None, f"invalid image-ID pin: {instance_id}")
        inspected = list(image_inspector(tag))
        require(len(inspected) == 1 and isinstance(inspected[0], Mapping), f"unexpected image inspection result: {tag}")
        local = inspected[0]
        require(local.get("Architecture") == "amd64", f"cached image architecture is not AMD64: {tag}")
        require(local.get("Id") == image_id, f"cached image ID differs from pin: {tag}")
        repo_digests = local.get("RepoDigests")
        require(
            isinstance(repo_digests, list)
            and all(isinstance(value, str) for value in repo_digests)
            and reference in repo_digests,
            f"cached image RepoDigest differs from pin: {tag}",
        )
        inspected_images.append(
            {
                "instance_id": instance_id,
                "tag": tag,
                "image_id": image_id,
                "repo_digests": sorted(repo_digests),
            }
        )
    return instance_ids, inspected_images


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()


def validate_selection_binding(
    proof: Mapping[str, Any],
    *,
    campaign_logical_path: str,
    campaign_sha256: str,
    instance_ids: Sequence[str],
    run_label: str,
    image_config_logical_path: str,
    image_config_sha256: str,
) -> str:
    """Validate a selected proof row; the CLI additionally requires the checker declaration."""

    expected_campaign_sha = EXPECTED_CAMPAIGNS.get(campaign_logical_path)
    require(
        expected_campaign_sha is not None and campaign_sha256 == expected_campaign_sha,
        "CONFIG is not one of the exact frozen V3 development campaigns",
    )
    require(run_label == EXPECTED_RUN_LABELS.get(campaign_logical_path), "RUN_ROOT label differs from the frozen V3 campaign binding")
    campaigns = mapping(proof.get("campaigns"), "selection proof campaigns")
    campaign_rows = [
        mapping(value, f"selection proof campaign {key}")
        for key, value in campaigns.items()
        if isinstance(value, dict) and "path" in value
    ]
    for expected_path, expected_sha in EXPECTED_CAMPAIGNS.items():
        matches = [row for row in campaign_rows if row.get("path") == expected_path]
        require(len(matches) == 1 and matches[0].get("sha256") == expected_sha, f"selection proof does not bind frozen campaign {expected_path}")
    selected_rows = [row for row in campaign_rows if row.get("path") == campaign_logical_path]
    require(len(selected_rows) == 1, "selection proof campaign binding is ambiguous")
    selected = selected_rows[0]
    require(
        selected.get("instance_count") == len(instance_ids)
        and selected.get("ordered_instance_ids_sha256") == _canonical_json_sha256(list(instance_ids))
        and selected.get("instance_ids_set_sha256") == _canonical_json_sha256(sorted(set(instance_ids))),
        "selection proof campaign task identity changed",
    )
    all_image_bindings = [mapping(row.get("image_registry"), "selection proof image registry") for row in campaign_rows]
    paths = [binding.get("path") for binding in all_image_bindings]
    require(all(isinstance(path, str) and path for path in paths) and len(paths) == len(set(paths)), "selection proof campaign image registry paths repeat")
    image_binding = mapping(selected.get("image_registry"), "selection proof campaign image registry")
    require(image_binding.get("generation_authorized") is True, "selection proof does not authorize generation")
    pinned_image_sha = image_binding.get("sha256")
    require(isinstance(pinned_image_sha, str) and re.fullmatch(r"[0-9a-f]{64}", pinned_image_sha) is not None, "selection proof image registry SHA-256 is not finalized")
    require(
        image_binding.get("path") == image_config_logical_path
        and pinned_image_sha == image_config_sha256,
        "image registry differs from the finalized selection-proof binding",
    )
    return pinned_image_sha


def protected_reserved_instance_ids(declaration: Any | None = None) -> frozenset[str]:
    """Return all 154 prior/reserved IDs from the checker-authenticated proof."""

    if declaration is None:
        declaration = _load_pinned_checker().validate_declaration()
    historical = mapping(declaration.proof.get("historical_exclusion"), "historical exclusion")
    raw_ids = historical.get("prior_reserved_instance_ids")
    require(isinstance(raw_ids, list), "prior/reserved instance IDs must be an array")
    instance_ids = list(raw_ids)
    require(
        historical.get("prior_reserved_instance_count") == len(instance_ids) == 154
        and len(set(instance_ids)) == 154
        and all(isinstance(value, str) and INSTANCE_ID_RE.fullmatch(value) for value in instance_ids),
        "checker-authenticated prior/reserved identity changed",
    )
    return frozenset(instance_ids)


def validate_run_root(
    path: Path,
    *,
    allowed_root: Path = V3_RUNS_ROOT,
    expected_label: str | None = None,
) -> tuple[Path, str]:
    """Require a new, exact direct child in the dedicated V3 namespace."""

    allowed = _lexical_absolute(allowed_root)
    candidate = _lexical_absolute(path)
    require(RUN_LABEL_RE.fullmatch(candidate.name) is not None, "unsafe V3 run label")
    require(candidate.parent == allowed, "V3 campaign run root must be a direct child of the dedicated development namespace")
    if expected_label is not None:
        require(candidate.name == expected_label, "RUN_ROOT differs from the exact declaration binding")
    require(not os.path.lexists(candidate), "V3 campaign run root must be new and absent")
    return candidate, candidate.name


def _open_directory_chain(path: Path) -> int:
    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for piece in absolute.parts[1:]:
            child = os.open(piece, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def securely_create_run_root(run_root: Path, *, namespace: Path = V3_RUNS_ROOT) -> Path:
    """Create a new empty run root using O_NOFOLLOW directory traversal."""

    candidate, run_label = validate_run_root(run_root, allowed_root=namespace)
    namespace = _lexical_absolute(namespace)
    try:
        parent_fd = _open_directory_chain(namespace.parent)
    except OSError as error:
        raise ValueError(f"V3 run-root path is unsafe: {error}") from error
    namespace_fd: int | None = None
    run_fd: int | None = None
    try:
        try:
            os.mkdir(namespace.name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        namespace_fd = os.open(namespace.name, flags, dir_fd=parent_fd)
        require(stat.S_ISDIR(os.fstat(namespace_fd).st_mode), "V3 run namespace is not a directory")
        try:
            os.mkdir(run_label, mode=0o755, dir_fd=namespace_fd)
        except FileExistsError as error:
            raise ValueError("V3 campaign run root must be new and absent") from error
        run_fd = os.open(run_label, flags, dir_fd=namespace_fd)
        require(stat.S_ISDIR(os.fstat(run_fd).st_mode), "new V3 run root is not a directory")
        require(not os.listdir(run_fd), "new V3 run root is not empty")
    except OSError as error:
        raise ValueError(f"V3 run-root path is unsafe: {error}") from error
    finally:
        if run_fd is not None:
            os.close(run_fd)
        if namespace_fd is not None:
            os.close(namespace_fd)
        os.close(parent_fd)
    require(candidate.is_dir() and not candidate.is_symlink(), "new V3 run root was not created safely")
    require(candidate.resolve(strict=True) == candidate, "new V3 run root traverses a symlink")
    return candidate


def atomic_write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    require(path.parent.is_dir() and not path.parent.is_symlink(), "JSON output parent is not a regular directory")
    require(path.parent.resolve(strict=True) == path.parent.absolute(), "JSON output parent traverses a symlink")
    require(not os.path.lexists(path), f"refusing to overwrite pre-existing output: {path.name}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and, unlike os.replace(), cannot
        # overwrite a target an adversary created after the preflight check.
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _declaration_campaign_index(declaration: Any, config_path: Path) -> int:
    matches = [index for index, path in enumerate(declaration.campaign_paths) if path == config_path]
    require(len(matches) == 1, "CONFIG is not one exact declaration campaign")
    return matches[0]


def prepare_campaign(
    *,
    config_path: Path,
    image_config_path: Path,
    selection_proof_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    # This declaration check is deliberately first: no local campaign output is
    # created, and no dataset/image work begins, until exact checked-in bytes pass.
    checker = _load_pinned_checker()
    require(Path(checker.V3_RUNS_ROOT) == V3_RUNS_ROOT, "V3 checker run namespace changed")
    declaration = checker.validate_declaration()
    rows = [mapping(value, f"cohort row {index}") for index, value in enumerate(declaration.cohort["cohorts"])]
    require(len(rows) == len(declaration.campaign_paths) == 2, "V3 declaration campaign count changed")

    requested_config = _lexical_absolute(config_path)
    campaign_index = _declaration_campaign_index(declaration, requested_config)
    row = rows[campaign_index]
    config_path = _exact_repository_file(config_path, declaration.campaign_paths[campaign_index], "campaign config")
    image_binding = mapping(row.get("image_registry"), "campaign image registry")
    expected_image_path = checker.repository_path(image_binding.get("path"), "campaign image registry path")
    image_config_path = _exact_repository_file(image_config_path, expected_image_path, "image registry")
    selection_proof_path = _exact_repository_file(selection_proof_path, declaration.proof_path, "selection proof")
    require(hashlib.sha256(selection_proof_path.read_bytes()).hexdigest() == EXPECTED_SELECTION_PROOF_SHA256, "selection proof SHA-256 changed")
    expected_label = str(row.get("run_label"))
    run_root, run_label = validate_run_root(run_root, expected_label=expected_label)

    campaign, campaign_bytes = strict_json_file(config_path, "campaign config")
    image_registry, image_registry_bytes = strict_json_file(image_config_path, "image registry")
    selection_proof, selection_proof_bytes = strict_json_file(selection_proof_path, "selection proof")
    campaign_sha256 = hashlib.sha256(campaign_bytes).hexdigest()
    image_config_sha256 = hashlib.sha256(image_registry_bytes).hexdigest()
    require(campaign == declaration.campaigns[campaign_index], "campaign bytes differ from declaration")
    require(selection_proof == declaration.proof, "selection proof bytes differ from declaration")
    require(campaign_sha256 == row.get("campaign_sha256"), "campaign SHA-256 differs from declaration")
    require(image_config_sha256 == image_binding.get("sha256"), "image registry SHA-256 differs from declaration")
    instance_ids, images = validate_campaign_contract(
        campaign,
        image_registry,
        forbidden_instance_ids=protected_reserved_instance_ids(declaration),
    )
    require(instance_ids == list(declaration.campaign_ids[campaign_index]), "campaign task order differs from declaration")
    validate_selection_binding(
        declaration.proof,
        campaign_logical_path=config_path.relative_to(ROOT).as_posix(),
        campaign_sha256=campaign_sha256,
        instance_ids=instance_ids,
        run_label=run_label,
        image_config_logical_path=image_config_path.relative_to(ROOT).as_posix(),
        image_config_sha256=image_config_sha256,
    )

    from datasets import load_dataset

    dataset = load_dataset(
        EXPECTED_DATASET["repo_id"],
        split="test",
        revision=EXPECTED_DATASET["revision"],
    )
    requested_ids = set(instance_ids)
    selected_by_id = {
        str(dataset_row["instance_id"]): dict(dataset_row)
        for dataset_row in dataset
        if dataset_row["instance_id"] in requested_ids
    }
    require(set(selected_by_id) == requested_ids, "frozen dataset revision does not contain the exact campaign cohort")
    selected = [selected_by_id[instance_id] for instance_id in instance_ids]

    run_root = securely_create_run_root(run_root)
    subset_path = run_root / "subset.json"
    dataset_path = run_root / "dataset.json"
    image_manifest_path = run_root / "image_manifest.json"
    selection_proof_sha256 = hashlib.sha256(selection_proof_bytes).hexdigest()
    atomic_write_json(
        subset_path,
        {"dataset_name": str(dataset_path.resolve(strict=False)), "instance_ids": instance_ids},
    )
    atomic_write_json(dataset_path, selected)
    atomic_write_json(
        image_manifest_path,
        {
            "schema_version": 1,
            "kind": IMAGE_MANIFEST_KIND,
            "campaign_config_path": str(config_path),
            "campaign_config_sha256": campaign_sha256,
            "image_config_path": str(image_config_path),
            "image_config_sha256": image_config_sha256,
            "selection_proof_path": str(selection_proof_path),
            "selection_proof_sha256": selection_proof_sha256,
            "dataset": EXPECTED_DATASET,
            "images": images,
        },
    )
    return {
        "run_root": str(run_root),
        "run_label": run_label,
        "task_count": len(instance_ids),
        "campaign_sha256": campaign_sha256,
        "image_config_sha256": image_config_sha256,
        "selection_proof_sha256": selection_proof_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-config", type=Path, required=True)
    parser.add_argument("--selection-proof", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_campaign(
        config_path=args.config,
        image_config_path=args.image_config,
        selection_proof_path=args.selection_proof,
        run_root=args.run_root,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
