from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_epistemic_chain_sealed_control_executor_v3.py"
)
SPEC = importlib.util.spec_from_file_location("sealed_control_executor_v3_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
executor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = executor
SPEC.loader.exec_module(executor)


def h(label: str) -> str:
    return executor.sha256_bytes(label.encode("utf-8"))


def authenticated_config():
    return executor.authenticate_executor_config(
        expected_config_sha256=executor.sha256_file(executor.CONFIG_PATH),
        expected_source_sha256=executor.sha256_file(SOURCE),
    )


def outer_launch_value(
    *,
    tmp_path: Path,
    role: str,
    adapter_config_sha256: str,
    adapter_source_sha256: str,
    adapter_launch_binding_sha256: str,
    request_batch_sha256: str,
) -> dict:
    generation = tmp_path / "outer-generation"
    keys = tmp_path / "outer-keys"
    generation.mkdir(exist_ok=True)
    keys.mkdir(exist_ok=True)
    config = authenticated_config()
    stage = "run_adjudicator" if role == "adjudicator" else "run_primary"
    authorization_nonce = h(f"{role}-outer-authorization-nonce")
    single_use_id = executor.sha256_value(
        {
            "domain": "v3-outer-launch-single-use-citrine",
            "suite_id": "development-control-suite-citrine-v3",
            "stage": stage,
            "role": role,
            "request_batch_sha256": request_batch_sha256,
            "authorization_nonce_sha256": authorization_nonce,
        }
    )
    return {
        "schema_version": executor.SCHEMA_VERSION,
        "interface_version": executor.INTERFACE_VERSION,
        "kind": executor.OUTER_LAUNCH_KIND,
        "suite_id": "development-control-suite-citrine-v3",
        "generation_root": str(generation.resolve()),
        "key_root": str(keys.resolve()),
        "stage": stage,
        "role": role,
        "execution_authorized": True,
        "model_access_authorized": True,
        "gpu_access_authorized": True,
        "artifact_output_authorized": True,
        "gate_evidence_authorized": True,
        "executor_config_sha256": config.config_sha256,
        "executor_source_sha256": config.source_sha256,
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_source_sha256": adapter_source_sha256,
        "adapter_launch_binding_sha256": adapter_launch_binding_sha256,
        "request_batch_sha256": request_batch_sha256,
        "suite_nonce_sha256": h("suite-nonce"),
        "nonce_precommit_receipt_sha256": h("nonce-precommit"),
        "prior_lock_sha256": h("dual-lock") if role == "adjudicator" else None,
        "authorization_nonce_sha256": authorization_nonce,
        "single_use_authorization_id": single_use_id,
        "retry_permitted": False,
        "reserved_validation_accessed": False,
    }


def rehash_trace(envelope: dict) -> dict:
    result = copy.deepcopy(envelope)
    previous = executor.ZERO_SHA256
    for ordinal, event in enumerate(result["trace"]["events"]):
        event["ordinal"] = ordinal
        event["previous_event_sha256"] = previous
        event["event_sha256"] = executor.sha256_value(
            {name: value for name, value in event.items() if name != "event_sha256"}
        )
        previous = event["event_sha256"]
    result["trace"]["event_count"] = len(result["trace"]["events"])
    result["trace"]["head_sha256"] = previous
    result["trace_sha256"] = executor.sha256_value(result["trace"])
    return result


def add_event(
    events: list[dict],
    *,
    stage: str,
    event_type: str,
    artifact_class: str,
    artifact_id: str,
    path: str | None = None,
    digest: str | None = None,
    clock: int,
) -> None:
    previous = events[-1]["event_sha256"] if events else executor.ZERO_SHA256
    events.append(
        executor.make_trace_event(
            ordinal=len(events),
            stage=stage,
            event_type=event_type,
            artifact_class=artifact_class,
            artifact_id=artifact_id,
            path=path,
            expected_sha256=digest if event_type == "read" else None,
            observed_sha256=(
                digest if event_type in {"read", "write", "consume"} else None
            ),
            monotonic_ns=clock,
            previous_event_sha256=previous,
        )
    )


def make_trace(
    *,
    suite_id: str,
    control_path: Path,
    fixture_path: Path,
    fixture_key_path: Path,
    nonce_secret_path: Path,
) -> dict:
    events: list[dict] = []
    clock = 1_000

    def add(**kwargs):
        nonlocal clock
        add_event(events, clock=clock, **kwargs)
        clock += 10

    add(
        stage="freeze_helper",
        event_type="read",
        artifact_class="protocol_config",
        artifact_id="executor_config",
        path=str(executor.CONFIG_PATH.resolve()),
        digest=executor.sha256_file(executor.CONFIG_PATH),
    )
    add(
        stage="freeze_helper",
        event_type="read",
        artifact_class="control_input",
        artifact_id="control_manifest_regenerated",
        path=str(control_path.resolve()),
        digest="cb2080a895cb219c8995e3944a0c86b5a0239e96fbde24086244a79de3049567",
    )
    add(
        stage="freeze_helper",
        event_type="read",
        artifact_class="fixture_input",
        artifact_id="fixture_manifest_regenerated",
        path=str(fixture_path.resolve()),
        digest="81338adf399e0835fc5030f79228f019dc3505893cc550a5957a1ac1346aef9e",
    )
    add(
        stage="freeze_helper",
        event_type="transition",
        artifact_class="stage",
        artifact_id="freeze_complete",
    )
    add(
        stage="precommit_nonce",
        event_type="transition",
        artifact_class="stage",
        artifact_id="nonce_precommitted",
    )
    add(
        stage="run_primary",
        event_type="read",
        artifact_class="control_input",
        artifact_id="control_manifest_regenerated_for_a",
        path=str(control_path.resolve()),
        digest="cb2080a895cb219c8995e3944a0c86b5a0239e96fbde24086244a79de3049567",
    )
    add(
        stage="run_primary",
        event_type="transition",
        artifact_class="stage",
        artifact_id="primary_independent_a_complete",
    )
    add(
        stage="run_primary",
        event_type="read",
        artifact_class="control_input",
        artifact_id="control_manifest_regenerated_for_b",
        path=str(control_path.resolve()),
        digest="cb2080a895cb219c8995e3944a0c86b5a0239e96fbde24086244a79de3049567",
    )
    add(
        stage="run_primary",
        event_type="transition",
        artifact_class="stage",
        artifact_id="primary_independent_b_complete",
    )
    add(
        stage="lock_primaries",
        event_type="transition",
        artifact_class="stage",
        artifact_id="dual_primary_lock_complete",
    )
    add(
        stage="run_adjudicator",
        event_type="read",
        artifact_class="control_input",
        artifact_id="control_manifest_regenerated_for_j",
        path=str(control_path.resolve()),
        digest="cb2080a895cb219c8995e3944a0c86b5a0239e96fbde24086244a79de3049567",
    )
    add(
        stage="run_adjudicator",
        event_type="read",
        artifact_class="fixture_input",
        artifact_id="fixture_manifest_regenerated_for_j",
        path=str(fixture_path.resolve()),
        digest="81338adf399e0835fc5030f79228f019dc3505893cc550a5957a1ac1346aef9e",
    )
    materializer = (
        ROOT
        / "scripts"
        / "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3.py"
    )
    add(
        stage="run_adjudicator",
        event_type="read",
        artifact_class="fixture_key_materializer",
        artifact_id="fixture_key_materializer_authenticated",
        path=str(materializer.resolve()),
        digest="9febe2a281acd2f20223db5dec5dd842365752df581ed522b2a628bc6e5f2172",
    )
    add(
        stage="run_adjudicator",
        event_type="read",
        artifact_class="fixture_generation_key",
        artifact_id="fixture_key_opened_once",
        path=str(fixture_key_path.resolve()),
        digest="d676af22f287e882400d3f356b49b921af3464b6b353477d6a04dedfb71e09cf",
    )
    add(
        stage="run_adjudicator",
        event_type="consume",
        artifact_class="nonce_secret",
        artifact_id="suite_nonce_consumed_once",
        path=str(nonce_secret_path.resolve()),
        digest=h("nonce-consumption-marker"),
    )
    add(
        stage="run_adjudicator",
        event_type="transition",
        artifact_class="stage",
        artifact_id="adjudicator_complete",
    )
    add(
        stage="lock_all",
        event_type="transition",
        artifact_class="stage",
        artifact_id="final_lock_complete",
    )
    add(
        stage="lock_all",
        event_type="transition",
        artifact_class="stage",
        artifact_id="trace_closed",
    )
    return executor.build_read_trace_envelope(
        suite_id=suite_id, events=events, closed=True
    )


def native_evidence(role: str, item_id: str) -> dict:
    return {
        "adapter_role": role,
        "adapter_config_sha256": h("adapter-config"),
        "adapter_source_sha256": h("adapter-source"),
        "outer_launch_authorization_sha256s": [h(f"{role}-{item_id}-outer-launch")],
        "launch_binding_sha256s": [h(f"{role}-{item_id}-launch")],
        "preflight_receipt_sha256s": [h(f"{role}-{item_id}-preflight")],
        "runtime_receipt_sha256s": [h(f"{role}-{item_id}-runtime")],
        "native_request_sha256s": [h(f"{role}-{item_id}-request")],
        "native_result_sha256s": [h(f"{role}-{item_id}-native-result")],
        "actual_model_execution": True,
        "model_loaded": True,
        "generation_performed": True,
        "gate_eligible": True,
    }


def role_manifest(role: str, packet_hashes: dict[str, str]) -> dict:
    rows = []
    for ordinal, control_id in enumerate(executor.CONTROL_IDS, start=1):
        result = {
            "decision": "no_chain" if control_id.startswith("C") else "novel"
        }
        bypass = control_id == "C32"
        rows.append(
            {
                "control_id": control_id,
                "ordinal": ordinal,
                "execution_path": (
                    "deterministic_host_bypass" if bypass else "native_model"
                ),
                "packet_sha256": packet_hashes[control_id],
                "result": result,
                "result_sha256": executor.sha256_value(result),
                "native_evidence": (
                    None if bypass else native_evidence(role, control_id)
                ),
            }
        )
    return {
        "role": role,
        "model_execution_count": 39,
        "host_bypass_count": 1,
        "result_count": 40,
        "ordered_results": rows,
        "role_receipt_sha256": h(f"{role}-receipt"),
    }


def materialize_input_files(generation_root: Path) -> tuple[Path, Path]:
    manifests = executor.regenerate_authoritative_inputs(authenticated_config())
    control_path = generation_root / "control-input.json"
    fixture_path = generation_root / "fixture-input.json"
    control_path.write_bytes(executor.canonical_json_bytes(manifests["control"]["manifest"]))
    fixture_path.write_bytes(executor.canonical_json_bytes(manifests["fixture"]["manifest"]))
    assert executor.sha256_file(control_path) == manifests["control"]["manifest_sha256"]
    assert executor.sha256_file(fixture_path) == manifests["fixture"]["manifest_sha256"]
    return control_path, fixture_path


def make_valid_artifacts(tmp_path: Path) -> tuple[dict, dict]:
    generation_root = tmp_path / "generation"
    key_root = tmp_path / "keys"
    generation_root.mkdir()
    key_root.mkdir()
    control_path, fixture_path = materialize_input_files(generation_root)
    control_input = executor._strict_json_bytes(
        control_path.read_bytes(), "test control input"
    )
    packet_hashes = {
        str(row["control_id"]): str(row["packet_sha256"])
        for row in (
            control_input["completion_records"] + control_input["novelty_records"]
        )
    }
    fixture_key_path = key_root / "fixture-key.json"
    fixture_key_path.write_bytes(
        (
            ROOT
            / "configs"
            / "swe_task_state_v4_epistemic_chain_adjudication_fixture_corpus_key_draft_v3.json"
        ).read_bytes()
    )
    assert (
        executor.sha256_file(fixture_key_path)
        == "d676af22f287e882400d3f356b49b921af3464b6b353477d6a04dedfb71e09cf"
    )
    nonce_secret_path = generation_root / "suite.nonce"
    nonce_secret_path.write_bytes(b"test-only-nonce-placeholder")
    suite_id = "development-control-suite-citrine-v3"
    trace = make_trace(
        suite_id=suite_id,
        control_path=control_path,
        fixture_path=fixture_path,
        fixture_key_path=fixture_key_path,
        nonce_secret_path=nonce_secret_path,
    )
    trace_external = executor.sha256_value(trace)
    executor.validate_read_trace_envelope(
        trace, independently_supplied_sha256=trace_external
    )
    schedule = [
        {"case_id": case_id, "verdict_seed": 1000 + ordinal, "repair_seed": 2000 + ordinal}
        for ordinal, case_id in enumerate(executor.FIXTURE_IDS, start=1)
    ]
    suite_nonce_sha256 = h("suite-nonce")
    fixture_rows = []
    for ordinal, case_id in enumerate(executor.FIXTURE_IDS, start=1):
        result = {
            "decision": (
                "no_chain"
                if executor.FIXTURE_PASSES[case_id] == "completion_chain"
                else "novel"
            )
        }
        fixture_rows.append(
            {
                "case_id": case_id,
                "ordinal": ordinal,
                "annotation_pass": executor.FIXTURE_PASSES[case_id],
                "result": result,
                "result_sha256": executor.sha256_value(result),
                "fixture_lock_sha256": h(f"{case_id}-fixture-lock"),
                "generation_contract_sha256": h(f"{case_id}-generation-contract"),
                "fixture_nonce_sha256": executor.derive_fixture_case_nonce_sha256(
                    suite_nonce_sha256=suite_nonce_sha256,
                    case_id=case_id,
                    fixture_input_manifest_sha256="81338adf399e0835fc5030f79228f019dc3505893cc550a5957a1ac1346aef9e",
                ),
                "verdict_seed": 1000 + ordinal,
                "repair_seed": 2000 + ordinal,
                "native_evidence": native_evidence("adjudicator", f"fixture-{case_id}"),
            }
        )
    role_a = role_manifest("independent_a", packet_hashes)
    role_b = role_manifest("independent_b", packet_hashes)
    role_j = role_manifest("adjudicator", packet_hashes)
    chronology = {
        "freeze_helper": {
            "stage": "freeze_helper",
            "receipt_sha256": h("freeze-receipt"),
            "completed_monotonic_ns": 100,
        },
        "precommit_nonce": {
            "stage": "precommit_nonce",
            "receipt_sha256": h("nonce-receipt"),
            "completed_monotonic_ns": 200,
        },
        "primary_roles": {
            "independent_a": {
                "stage": "run_primary",
                "receipt_sha256": role_a["role_receipt_sha256"],
                "completed_monotonic_ns": 300,
            },
            "independent_b": {
                "stage": "run_primary",
                "receipt_sha256": role_b["role_receipt_sha256"],
                "completed_monotonic_ns": 310,
            },
        },
        "lock_primaries": {
            "stage": "lock_primaries",
            "receipt_sha256": h("dual-lock"),
            "completed_monotonic_ns": 400,
        },
        "adjudicator": {
            "stage": "run_adjudicator",
            "receipt_sha256": role_j["role_receipt_sha256"],
            "completed_monotonic_ns": 500,
        },
        "lock_all": {
            "stage": "lock_all",
            "receipt_sha256": h("final-lock"),
            "completed_monotonic_ns": 600,
        },
    }
    manifest = {
        "schema_version": executor.SCHEMA_VERSION,
        "interface_version": executor.INTERFACE_VERSION,
        "kind": executor.PUBLIC_BUNDLE_KIND,
        "status": "all_generation_outputs_locked_scoring_keys_unread",
        "scope": copy.deepcopy(executor.EXPECTED_SCOPE),
        "suite_id": suite_id,
        "executor_identity": {
            "config_path": str(executor.CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": executor.sha256_file(executor.CONFIG_PATH),
            "source_path": str(SOURCE.relative_to(ROOT)),
            "source_sha256": executor.sha256_file(SOURCE),
            "adapter_config_sha256": h("adapter-config"),
            "adapter_source_sha256": h("adapter-source"),
        },
        "suite_nonce": {
            "suite_nonce_sha256": suite_nonce_sha256,
            "precommit_receipt_sha256": chronology["precommit_nonce"]["receipt_sha256"],
            "single_use_consumption_receipt_sha256": h("nonce-consumption"),
            "retry_permitted": False,
            "raw_nonce_public": False,
        },
        "filesystem_roots": {
            "generation_root": str(generation_root.resolve()),
            "key_root": str(key_root.resolve()),
        },
        "inputs": {
            "control": {
                "path": str(control_path.resolve()),
                "sha256": executor.sha256_file(control_path),
            },
            "fixture": {
                "path": str(fixture_path.resolve()),
                "sha256": executor.sha256_file(fixture_path),
            },
        },
        "key_commitments": {
            "fixture_generation_key": {
                "sha256": "d676af22f287e882400d3f356b49b921af3464b6b353477d6a04dedfb71e09cf",
                "read_by_executor": True,
                "first_read_stage": "run_adjudicator",
                "read_count": 1,
            },
            "control_scoring_key": {
                "sha256": "806da55baf4f39f18f7835d061f8729e2a55b8968a80abcb719960c699ac8250",
                "read_by_executor": False,
                "scorer_reads_last": True,
            },
        },
        "chronology": chronology,
        "roles": {
            "independent_a": role_a,
            "independent_b": role_b,
            "adjudicator": role_j,
        },
        "fixture_results": fixture_rows,
        "fixture_seed_schedule": schedule,
        "fixture_seed_schedule_sha256": executor.sha256_value(schedule),
        "locks": {
            "primary_independent_a_sha256": role_a["role_receipt_sha256"],
            "primary_independent_b_sha256": role_b["role_receipt_sha256"],
            "dual_primary_lock_sha256": chronology["lock_primaries"]["receipt_sha256"],
            "adjudicator_sha256": role_j["role_receipt_sha256"],
            "final_lock_sha256": chronology["lock_all"]["receipt_sha256"],
        },
        "read_trace": {
            "trace_envelope_sha256": trace_external,
            "trace_sha256": trace["trace_sha256"],
            "head_sha256": trace["trace"]["head_sha256"],
            "event_count": trace["trace"]["event_count"],
        },
        "claims": copy.deepcopy(executor.PUBLIC_CLAIMS),
    }
    bundle = {"manifest": manifest, "manifest_sha256": executor.sha256_value(manifest)}
    return trace, bundle


def rehash_bundle(bundle: dict) -> dict:
    result = copy.deepcopy(bundle)
    result["manifest_sha256"] = executor.sha256_value(result["manifest"])
    return result


def test_executor_config_authenticates_and_is_all_false():
    config = authenticated_config()
    assert all(
        config.value["authorization"][name] is False
        for name in (
            "execution_authorized",
            "model_access_authorized",
            "gpu_access_authorized",
            "artifact_output_authorized",
            "gate_evidence_authorized",
        )
    )


def test_freeze_helper_regenerates_exact_inputs_without_run_claim():
    result = executor.freeze_helper(
        expected_config_sha256=executor.sha256_file(executor.CONFIG_PATH),
        expected_source_sha256=executor.sha256_file(SOURCE),
    )
    assert result["receipt"]["control_input_manifest_sha256"].startswith("cb2080")
    assert result["receipt"]["fixture_input_manifest_sha256"].startswith("81338a")
    assert result["receipt"]["claims"] == {
        "model_or_gpu_execution_performed": False,
        "persistent_artifact_written": False,
        "fixture_key_read": False,
        "control_scoring_key_read": False,
        "sealed_run_claimed": False,
    }


def test_adapter_delegation_rejects_missing_launch_before_request_or_adapter_access():
    class Poison:
        def __iter__(self):
            raise AssertionError("request specs accessed")

    adapter_config = (
        ROOT / "configs" / "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.json"
    )
    adapter_source = (
        ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_native_vllm_adapter_v3.py"
    )
    with pytest.raises(executor.SealedControlExecutorError, match="outer launch"):
        executor.delegate_native_adapter_batch(
            authenticated_executor_config=authenticated_config(),
            outer_launch_authorization_path=Path("missing-outer-launch.json"),
            expected_outer_launch_authorization_sha256=h("outer-launch"),
            outer_authorization_consumption_marker_path=Path(
                "must-not-create-consumption-marker.json"
            ),
            expected_adapter_config_sha256=executor.sha256_file(adapter_config),
            expected_adapter_source_sha256=executor.sha256_file(adapter_source),
            launch_binding_path=Path("missing-launch.json"),
            expected_launch_binding_sha256=h("launch"),
            role="independent_a",
            request_specs=Poison(),
        )


def test_outer_launch_is_distinct_true_hash_bound_authority(tmp_path: Path):
    adapter_config_sha = h("adapter-config")
    adapter_source_sha = h("adapter-source")
    adapter_launch_sha = h("adapter-launch")
    request_batch_sha = h("request-batch")
    value = outer_launch_value(
        tmp_path=tmp_path,
        role="independent_a",
        adapter_config_sha256=adapter_config_sha,
        adapter_source_sha256=adapter_source_sha,
        adapter_launch_binding_sha256=adapter_launch_sha,
        request_batch_sha256=request_batch_sha,
    )
    path = tmp_path / "outer-launch.json"
    path.write_bytes(executor.canonical_json_bytes(value))
    observed = executor._precheck_outer_launch_authorization(
        path=path,
        expected_sha256=executor.sha256_file(path),
        authenticated_executor_config=authenticated_config(),
        adapter_config_sha256=adapter_config_sha,
        adapter_source_sha256=adapter_source_sha,
        adapter_launch_binding_sha256=adapter_launch_sha,
        role="independent_a",
    )
    assert observed["execution_authorized"] is True
    assert authenticated_config().value["authorization"]["execution_authorized"] is False
    tampered = copy.deepcopy(value)
    tampered["execution_authorized"] = False
    bad_path = tmp_path / "outer-launch-false.json"
    bad_path.write_bytes(executor.canonical_json_bytes(tampered))
    with pytest.raises(executor.SealedControlExecutorError, match="does not authorize"):
        executor._precheck_outer_launch_authorization(
            path=bad_path,
            expected_sha256=executor.sha256_file(bad_path),
            authenticated_executor_config=authenticated_config(),
            adapter_config_sha256=adapter_config_sha,
            adapter_source_sha256=adapter_source_sha,
            adapter_launch_binding_sha256=adapter_launch_sha,
            role="independent_a",
        )


@pytest.mark.parametrize("attack", ["retry", "single_use_rebind"])
def test_outer_launch_rejects_retry_or_single_use_rebinding(
    tmp_path: Path, attack: str
):
    adapter_config_sha = h("adapter-config")
    adapter_source_sha = h("adapter-source")
    adapter_launch_sha = h("adapter-launch")
    value = outer_launch_value(
        tmp_path=tmp_path,
        role="independent_a",
        adapter_config_sha256=adapter_config_sha,
        adapter_source_sha256=adapter_source_sha,
        adapter_launch_binding_sha256=adapter_launch_sha,
        request_batch_sha256=h("request-batch"),
    )
    if attack == "retry":
        value["retry_permitted"] = True
    else:
        value["single_use_authorization_id"] = h("attacker-single-use-id")
    path = tmp_path / f"outer-launch-{attack}.json"
    path.write_bytes(executor.canonical_json_bytes(value))
    with pytest.raises(executor.SealedControlExecutorError):
        executor._precheck_outer_launch_authorization(
            path=path,
            expected_sha256=executor.sha256_file(path),
            authenticated_executor_config=authenticated_config(),
            adapter_config_sha256=adapter_config_sha,
            adapter_source_sha256=adapter_source_sha,
            adapter_launch_binding_sha256=adapter_launch_sha,
            role="independent_a",
        )


def test_disjoint_roots_reject_equal_or_nested(tmp_path: Path):
    generation = tmp_path / "generation"
    generation.mkdir()
    nested = generation / "keys"
    nested.mkdir()
    with pytest.raises(executor.SealedControlExecutorError, match="disjoint"):
        executor.assert_disjoint_roots(generation_root=generation, key_root=nested)


def test_disjoint_roots_reject_symlink_alias(tmp_path: Path):
    generation = tmp_path / "generation"
    keys = tmp_path / "keys"
    generation.mkdir()
    keys.mkdir()
    alias = tmp_path / "generation-alias"
    alias.symlink_to(generation, target_is_directory=True)
    with pytest.raises(executor.SealedControlExecutorError, match="symlink"):
        executor.assert_disjoint_roots(generation_root=alias, key_root=keys)


def test_precommit_and_consumption_are_single_use(tmp_path: Path):
    generation = tmp_path / "generation"
    keys = tmp_path / "keys"
    generation.mkdir()
    keys.mkdir()
    kwargs = {
        "suite_id": "development-suite-single-use",
        "generation_root": generation,
        "key_root": keys,
        "nonce_secret_path": generation / "nonce.bin",
        "single_use_marker_path": generation / "nonce.claim",
        "receipt_path": generation / "nonce-receipt.json",
        "freeze_receipt_sha256": h("freeze"),
    }
    first = executor.precommit_suite_nonce(**kwargs)
    assert first["receipt"]["raw_nonce_public"] is False
    with pytest.raises(executor.SealedControlExecutorError, match="retry forbidden"):
        executor.precommit_suite_nonce(**kwargs)
    consumed = executor.consume_suite_nonce_once(
        suite_id=kwargs["suite_id"],
        generation_root=generation,
        key_root=keys,
        nonce_secret_path=kwargs["nonce_secret_path"],
        expected_nonce_secret_file_sha256=first["receipt"]["nonce_secret_file_sha256"],
        expected_suite_nonce_sha256=first["receipt"]["suite_nonce_sha256"],
        consumption_marker_path=generation / "nonce.consumed",
        dual_primary_lock_sha256=h("dual-lock"),
    )
    assert consumed["marker"]["consumed_once"] is True
    with pytest.raises(executor.SealedControlExecutorError, match="already exists"):
        executor.consume_suite_nonce_once(
            suite_id=kwargs["suite_id"],
            generation_root=generation,
            key_root=keys,
            nonce_secret_path=kwargs["nonce_secret_path"],
            expected_nonce_secret_file_sha256=first["receipt"]["nonce_secret_file_sha256"],
            expected_suite_nonce_sha256=first["receipt"]["suite_nonce_sha256"],
            consumption_marker_path=generation / "nonce.consumed",
            dual_primary_lock_sha256=h("dual-lock"),
        )


def test_closed_trace_and_public_bundle_validate(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    authenticated_trace = executor.validate_read_trace_envelope(
        trace, independently_supplied_sha256=executor.sha256_value(trace)
    )
    manifest = executor.validate_public_generation_bundle(
        bundle,
        independently_supplied_sha256=executor.sha256_value(bundle),
        authenticated_trace=authenticated_trace,
    )
    assert manifest["status"] == "all_generation_outputs_locked_scoring_keys_unread"


def test_trace_journal_is_exclusive_append_only_and_closes(tmp_path: Path):
    trace, _ = make_valid_artifacts(tmp_path)
    journal = tmp_path / "trace.jsonl"
    digest = executor.create_read_trace_journal(journal)
    with pytest.raises(executor.SealedControlExecutorError, match="already exists"):
        executor.create_read_trace_journal(journal)
    for source_event in trace["trace"]["events"]:
        appended = executor.append_read_trace_event(
            journal_path=journal,
            expected_journal_file_sha256=digest,
            stage=source_event["stage"],
            event_type=source_event["event_type"],
            artifact_class=source_event["artifact_class"],
            artifact_id=source_event["artifact_id"],
            path=source_event["path"],
            expected_sha256=source_event["expected_sha256"],
            observed_sha256=source_event["observed_sha256"],
            monotonic_ns=source_event["monotonic_ns"],
        )
        digest = appended["journal_file_sha256"]
    closed = executor.close_read_trace_journal(
        journal_path=journal,
        expected_journal_file_sha256=digest,
        suite_id=trace["trace"]["suite_id"],
        output_envelope_path=tmp_path / "closed-trace.json",
    )
    assert closed["envelope"]["trace"]["closed"] is True
    with pytest.raises(executor.SealedControlExecutorError, match="already closed"):
        executor.append_read_trace_event(
            journal_path=journal,
            expected_journal_file_sha256=digest,
            stage="lock_all",
            event_type="transition",
            artifact_class="stage",
            artifact_id="late-replay",
            path=None,
            expected_sha256=None,
            observed_sha256=None,
        )


def test_trace_coherent_tamper_and_inner_rehash_fails_external_root(tmp_path: Path):
    trace, _ = make_valid_artifacts(tmp_path)
    expected = executor.sha256_value(trace)
    tampered = copy.deepcopy(trace)
    tampered["trace"]["suite_id"] += "-tampered"
    tampered["trace_sha256"] = executor.sha256_value(tampered["trace"])
    with pytest.raises(executor.SealedControlExecutorError, match="independently"):
        executor.validate_read_trace_envelope(
            tampered, independently_supplied_sha256=expected
        )


def test_trace_rejects_key_read_before_dual_lock_even_if_rehashed(tmp_path: Path):
    trace, _ = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(trace)
    event = next(
        item
        for item in tampered["trace"]["events"]
        if item["artifact_class"] == "fixture_generation_key"
    )
    event["stage"] = "run_primary"
    event["stage_rank"] = 2
    tampered = rehash_trace(tampered)
    with pytest.raises(executor.SealedControlExecutorError):
        executor.validate_read_trace_envelope(
            tampered, independently_supplied_sha256=executor.sha256_value(tampered)
        )


def test_trace_rejects_nonce_replay_even_if_chain_rehashed(tmp_path: Path):
    trace, _ = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(trace)
    consume = next(
        item
        for item in tampered["trace"]["events"]
        if item["event_type"] == "consume"
    )
    duplicate = copy.deepcopy(consume)
    index = tampered["trace"]["events"].index(consume)
    duplicate["monotonic_ns"] += 1
    tampered["trace"]["events"].insert(index + 1, duplicate)
    tampered = rehash_trace(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="consumed exactly once"):
        executor.validate_read_trace_envelope(
            tampered, independently_supplied_sha256=executor.sha256_value(tampered)
        )


def test_trace_rejects_chronology_regression_after_coherent_rehash(tmp_path: Path):
    trace, _ = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(trace)
    dual_index = next(
        index
        for index, event in enumerate(tampered["trace"]["events"])
        if event["artifact_id"] == "dual_primary_lock_complete"
    )
    primary_index = next(
        index
        for index, event in enumerate(tampered["trace"]["events"])
        if event["artifact_id"] == "primary_independent_b_complete"
    )
    primary = tampered["trace"]["events"].pop(primary_index)
    tampered["trace"]["events"].insert(dual_index + 1, primary)
    tampered = rehash_trace(tampered)
    with pytest.raises(executor.SealedControlExecutorError):
        executor.validate_read_trace_envelope(
            tampered, independently_supplied_sha256=executor.sha256_value(tampered)
        )


def test_bundle_coherent_result_tamper_fails_original_external_root(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    expected = executor.sha256_value(bundle)
    tampered = copy.deepcopy(bundle)
    row = tampered["manifest"]["roles"]["independent_a"]["ordered_results"][0]
    row["result"] = {"decision": "chain"}
    row["result_sha256"] = executor.sha256_value(row["result"])
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="independently"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=expected,
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_co_tampered_executor_identity_after_full_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["manifest"]["executor_identity"]["config_sha256"] = h(
        "attacker-config"
    )
    tampered["manifest"]["executor_identity"]["source_sha256"] = h(
        "attacker-source"
    )
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="executor identity"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_extra_host_bypass_with_fully_rehashed_envelope(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    lane = tampered["manifest"]["roles"]["independent_a"]
    row = lane["ordered_results"][0]
    row["execution_path"] = "deterministic_host_bypass"
    row["native_evidence"] = None
    lane["model_execution_count"] = 38
    lane["host_bypass_count"] = 2
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_c32_native_model_path_with_fully_rehashed_envelope(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    row = tampered["manifest"]["roles"]["adjudicator"]["ordered_results"][31]
    row["execution_path"] = "native_model"
    row["native_evidence"] = native_evidence("adjudicator", "C32")
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="C32"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_wrong_pass_semantics_after_full_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    novelty = tampered["manifest"]["roles"]["independent_a"]["ordered_results"][32]
    novelty["result"] = {"decision": "no_chain"}
    novelty["result_sha256"] = executor.sha256_value(novelty["result"])
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="novelty decision"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_forged_chain_ids_after_full_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    row = tampered["manifest"]["roles"]["independent_a"]["ordered_results"][0]
    row["result"] = {
        "decision": "chain",
        "evidence_unit_id": "forged-evidence-unit",
        "hypothesis_unit_id": "forged-hypothesis-unit",
        "action_unit_id": "forged-action-unit",
        "evidence_kind": "code",
        "belief_edge": "supports",
        "hypothesis_domain": "source_logic",
        "action_intent": "inspect",
    }
    row["result_sha256"] = executor.sha256_value(row["result"])
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="E<H<A"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_replayed_native_result_with_fully_rehashed_envelope(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    rows = tampered["manifest"]["roles"]["independent_b"]["ordered_results"]
    rows[1]["native_evidence"]["native_result_sha256s"] = copy.deepcopy(
        rows[0]["native_evidence"]["native_result_sha256s"]
    )
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="replayed"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_cross_role_native_replay_after_full_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    roles = tampered["manifest"]["roles"]
    roles["independent_b"]["ordered_results"][0]["native_evidence"][
        "native_result_sha256s"
    ] = copy.deepcopy(
        roles["independent_a"]["ordered_results"][0]["native_evidence"][
            "native_result_sha256s"
        ]
    )
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="finalized bundle"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_monotonic_chronology_attack_after_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["manifest"]["chronology"]["adjudicator"]["completed_monotonic_ns"] = 350
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="chronology"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_seed_schedule_tamper_after_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["manifest"]["fixture_seed_schedule"][0]["verdict_seed"] += 1
    tampered["manifest"]["fixture_seed_schedule_sha256"] = executor.sha256_value(
        tampered["manifest"]["fixture_seed_schedule"]
    )
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="seed"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_fixture_key_commitment_tamper_after_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["manifest"]["key_commitments"]["fixture_generation_key"]["sha256"] = h(
        "replacement-key"
    )
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="key commitment"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_input_key_root_overlap_after_rehash(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["manifest"]["filesystem_roots"]["key_root"] = tampered["manifest"][
        "filesystem_roots"
    ]["generation_root"]
    tampered = rehash_bundle(tampered)
    with pytest.raises(executor.SealedControlExecutorError, match="disjoint"):
        executor.validate_public_generation_bundle(
            tampered,
            independently_supplied_sha256=executor.sha256_value(tampered),
            authenticated_trace=trace["trace"],
        )


def test_bundle_rejects_missing_authoritative_primary_reread(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    tampered_trace = copy.deepcopy(trace)
    events = tampered_trace["trace"]["events"]
    index = next(
        i
        for i, event in enumerate(events)
        if event["artifact_id"] == "control_manifest_regenerated_for_b"
    )
    events.pop(index)
    tampered_trace = rehash_trace(tampered_trace)
    authenticated_trace = executor.validate_read_trace_envelope(
        tampered_trace,
        independently_supplied_sha256=executor.sha256_value(tampered_trace),
    )
    tampered_bundle = copy.deepcopy(bundle)
    tampered_bundle["manifest"]["read_trace"] = {
        "trace_envelope_sha256": executor.sha256_value(tampered_trace),
        "trace_sha256": tampered_trace["trace_sha256"],
        "head_sha256": tampered_trace["trace"]["head_sha256"],
        "event_count": tampered_trace["trace"]["event_count"],
    }
    tampered_bundle = rehash_bundle(tampered_bundle)
    # A set-only check would miss one primary lane.  There must be two distinct
    # run_primary input regenerations, one before each primary completion.
    with pytest.raises(executor.SealedControlExecutorError):
        executor.validate_public_generation_bundle(
            tampered_bundle,
            independently_supplied_sha256=executor.sha256_value(tampered_bundle),
            authenticated_trace=authenticated_trace,
        )


def test_duplicate_json_keys_rejected_for_cli_input(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    digest = executor.sha256_file(path)
    with pytest.raises(executor.SealedControlExecutorError, match="duplicate"):
        executor._load_cli_json(path, digest, "duplicate test")


def test_runtime_claim_constant_mutation_fails_closed(tmp_path: Path):
    trace, bundle = make_valid_artifacts(tmp_path)
    original = copy.deepcopy(executor.PUBLIC_CLAIMS)
    try:
        executor.PUBLIC_CLAIMS["actual_model_execution_authenticated"] = False
        with pytest.raises(executor.SealedControlExecutorError, match="mutated"):
            executor.validate_public_generation_bundle(
                bundle,
                independently_supplied_sha256=executor.sha256_value(bundle),
                authenticated_trace=trace["trace"],
            )
    finally:
        executor.PUBLIC_CLAIMS.clear()
        executor.PUBLIC_CLAIMS.update(original)
