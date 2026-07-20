from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_native_smoke_v3.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = load_module("native_smoke_v3_test", SOURCE)
adapter = smoke.adapter


def h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def authenticated_config():
    return adapter.AuthenticatedAdapterConfig(
        value=copy.deepcopy(
            adapter.load_adapter_config(
                path=adapter.CONFIG_PATH,
                expected_config_sha256=adapter.sha256_file(adapter.CONFIG_PATH),
            )
        ),
        path=adapter.CONFIG_PATH,
        config_sha256=adapter.sha256_file(adapter.CONFIG_PATH),
        source_sha256=adapter.sha256_file(adapter.SOURCE_PATH),
        runner_sha256=h("runner"),
        draft_config_sha256=h("draft-config"),
        draft_source_sha256=h("draft-source"),
        v2_config_sha256=h("v2-config"),
    )


def test_expected_gpu_identity_is_finite_and_self_hashed():
    value = smoke.expected_gpu_identity()
    body = {key: item for key, item in value.items() if key != "gpu_identity_sha256"}
    assert value["gpu_identity_sha256"] == smoke.sha256_value(body)
    assert value["device_name"] == "NVIDIA GeForce RTX 5090"
    assert value["compute_capability"] == [12, 0]
    assert value["total_memory_bytes"] == 33635434496


def test_launch_builder_emits_exact_adapter_field_set_and_true_shell():
    authenticated = authenticated_config()
    launch = smoke.build_launch_authorization(
        authenticated_config=authenticated,
        role="independent_a",
        request_batch_sha256=h("request-batch"),
        runtime_identity={"runtime_identity_sha256": h("runtime")},
        environment_identity={"environment_identity_sha256": h("environment")},
        package_identity={"package_bundle_sha256": h("packages")},
        snapshot_inventory={"inventory_sha256": h("snapshot")},
        authorization_nonce_sha256=h("nonce"),
    )
    assert set(launch) == adapter.LAUNCH_FIELDS
    for name in (
        "execution_authorized",
        "model_access_authorized",
        "gpu_access_authorized",
        "output_authorized",
        "production_receipt_authorized",
        "gate_eligible_execution_authorized",
    ):
        assert launch[name] is True
    assert launch["request_batch_sha256"] == h("request-batch")
    assert launch["role"] == "independent_a"


def test_smoke_schema_has_only_one_finite_output():
    schema = smoke.smoke_schema()
    assert schema == {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["ok"]}},
        "required": ["verdict"],
        "additionalProperties": False,
    }
    adapter.runner_v3.validate_executable_response_schema(schema)


def test_freeze_path_has_no_torch_import_or_gpu_query():
    source = inspect.getsource(smoke.freeze_smoke)
    context_source = inspect.getsource(smoke._load_verified_tokenizer_context)
    combined = source + context_source
    assert "import torch" not in combined
    assert "torch.cuda" not in combined
    assert "adapter._gpu_identity(" not in combined
    assert "get_device" not in combined


def test_receipt_claim_literals_do_not_promote_smoke_to_cot_or_affect():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"sealed_control_evidence_established": False' in source
    assert '"private_or_verbatim_cot_recovery_established": False' in source
    assert '"latent_cot_like_trajectory_recovery_established": False' in source
    assert (
        '"affect_emotion_confidence_doubt_or_stress_recovery_established": False'
        in source
    )
    assert '"reserved_validation_accessed": False' in source


def test_strict_json_rejects_duplicate_keys():
    raw = b'{"x":1,"x":2}'
    try:
        smoke._strict_json_bytes(raw, "attack")
    except smoke.NativeSmokeError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate JSON key accepted")


def test_canonical_json_has_no_nonfinite_or_whitespace_variance():
    assert smoke.canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    try:
        smoke.canonical_json_bytes({"x": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite JSON accepted")


def test_current_three_role_index_authenticates_final_receipts_when_present():
    index_path = (
        ROOT
        / "configs"
        / "swe_task_state_v4_epistemic_chain_native_smoke_current_v3.json"
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["status"] == (
        "three_role_native_exact_token_smoke_complete_not_sealed_control_evidence"
    )
    assert index["bindings"]["adapter_source"]["sha256"] == smoke.sha256_file(
        ROOT / index["bindings"]["adapter_source"]["path"]
    )
    assert index["bindings"]["adapter_config"]["sha256"] == smoke.sha256_file(
        ROOT / index["bindings"]["adapter_config"]["path"]
    )
    assert index["bindings"]["smoke_harness"]["sha256"] == smoke.sha256_file(
        ROOT / index["bindings"]["smoke_harness"]["path"]
    )
    artifact_root = ROOT / index["bindings"]["artifact_root"]
    for role, expected in index["roles"].items():
        receipt_path = artifact_root / role / "smoke-receipt.json"
        assert receipt_path.is_file()
        assert smoke.sha256_file(receipt_path) == expected["smoke_receipt_file_sha256"]
        envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt = envelope["receipt"]
        assert smoke.sha256_value(receipt) == envelope["receipt_sha256"]
        assert envelope["receipt_sha256"] == expected["smoke_receipt_sha256"]
        assert receipt["role"] == role
        assert receipt["native_request_sha256"] == expected["native_request_sha256"]
        assert receipt["native_result_sha256"] == expected["native_result_sha256"]
        assert (
            receipt["preflight_receipt_sha256"] == expected["preflight_receipt_sha256"]
        )
        assert receipt["runtime_receipt_sha256"] == expected["runtime_receipt_sha256"]
        runtime_record = receipt["runtime_receipt"]["request_records"][0]
        assert runtime_record["engine_prompt_matches_submitted"] is True
        assert runtime_record["candidate_text_token_decode_parity"] is True
        assert receipt["runtime_receipt"]["snapshot_unchanged"] is True
        harmony = runtime_record["output_extraction"].get("harmony")
        recipient = None if harmony is None else harmony["final_recipient"]
        assert recipient == expected["harmony_final_recipient"]
        assert receipt["claims"]["sealed_control_evidence_established"] is False
        assert (
            receipt["claims"]["latent_cot_like_trajectory_recovery_established"]
            is False
        )
        assert (
            receipt["claims"][
                "affect_emotion_confidence_doubt_or_stress_recovery_established"
            ]
            is False
        )
