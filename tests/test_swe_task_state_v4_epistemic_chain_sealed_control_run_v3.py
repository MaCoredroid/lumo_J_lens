from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_epistemic_chain_sealed_control_run_v3.py"
)
SPEC = importlib.util.spec_from_file_location("sealed_control_run_v3_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
control_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control_run
SPEC.loader.exec_module(control_run)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        digest = control_run.sha256_value(list(messages))
        return [11, int(digest[:8], 16), len(messages), 12]


def fake_context():
    model = {
        "base_model_lineage": "test-lineage",
        "repo_id": "test/model",
        "revision": "test-revision",
        "snapshot_tree_sha256": control_run.sha256_value("snapshot"),
        "quantization": "test",
        "dtype": "bfloat16",
    }
    tokenizer = {
        "repo_id": "test/model",
        "revision": "test-revision",
        "snapshot_tree_sha256": model["snapshot_tree_sha256"],
        "tokenizer_mode": "auto",
        "tokenizer_class": "tests.FakeTokenizer",
        "vocab_identity_sha256": control_run.sha256_value("vocab"),
    }
    return control_run.runner.authenticate_native_generation_context(
        tokenizer=FakeTokenizer(),
        model_identity=model,
        expected_model_identity_sha256=control_run.sha256_value(model),
        tokenizer_identity=tokenizer,
        expected_tokenizer_identity_sha256=control_run.sha256_value(tokenizer),
        chat_template_kwargs={},
    )


def control_manifest():
    authenticated = control_run.executor.authenticate_executor_config(
        expected_config_sha256=control_run.executor.sha256_file(
            control_run.executor.CONFIG_PATH
        ),
        expected_source_sha256=control_run.executor.sha256_file(
            control_run.executor.SOURCE_PATH
        ),
    )
    return control_run.executor.regenerate_authoritative_inputs(authenticated)[
        "control"
    ]["manifest"]


def request_from_spec(spec, context):
    return control_run.runner.build_native_generation_request(
        context=context,
        messages=spec.messages,
        schema=spec.schema,
        seed=spec.seed,
        stage=spec.stage,
        annotation_pass=spec.annotation_pass,
        packet_id_sha256=spec.packet_id_sha256,
        source_id_sha256=spec.source_id_sha256,
        lineage_bindings=spec.lineage_bindings,
    )


def batch_envelope(*, specs, context, chain_packet_ids=frozenset()):
    requests = [request_from_spec(spec, context) for spec in specs]
    results = []
    for request in requests:
        if request.body["annotation_pass"] == "prefix_novelty":
            text = '{"decision":"novel"}'
        elif request.body["packet_id_sha256"] in chain_packet_ids:
            text = '{"decision":"chain"}'
        else:
            text = '{"decision":"no_chain"}'
        ids = request.body["submitted_prompt_token_ids"]
        results.append(
            control_run.runner.build_native_generation_result(
                request=request,
                text=text,
                submitted_prompt_token_ids=ids,
                engine_prompt_token_ids=ids,
                output_token_ids=[91],
                finish_reason="stop",
            )
        )
    receipt_body = {"test": True}
    artifact = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": control_run.BATCH_ARTIFACT_KIND,
        "status": "authenticated_native_batch_complete",
        "suite_id": "test-development-suite-citrine-v3",
        "role": "independent_a",
        "round": "initial",
        "freeze_manifest_file_sha256": control_run.sha256_value("freeze"),
        "adapter_launch_file_sha256": control_run.sha256_value("adapter-launch"),
        "outer_launch_file_sha256": control_run.sha256_value("outer-launch"),
        "request_count": len(requests),
        "requests": [asdict(item) for item in requests],
        "results": [asdict(item) for item in results],
        "preflight_receipt": {
            "body": receipt_body,
            "receipt_sha256": control_run.sha256_value(receipt_body),
        },
        "runtime_receipt": {
            "body": receipt_body,
            "receipt_sha256": control_run.sha256_value(receipt_body),
        },
        "completed_monotonic_ns": 100,
        "claims": {
            "actual_model_execution": True,
            "gate_eligible": True,
            "reserved_validation_accessed": False,
            "private_or_verbatim_cot_recovery_established": False,
            "affect_emotion_confidence_doubt_or_stress_established": False,
        },
    }
    return {"artifact": artifact, "artifact_sha256": control_run.sha256_value(artifact)}


def test_primary_initial_round_is_exactly_39_and_not_scheduler_capped():
    context = fake_context()
    specs = control_run.build_primary_initial_specs(
        role="independent_a",
        control_manifest=control_manifest(),
        context=context,
        seed=17011,
    )
    assert len(specs) == 39
    assert sum(item.annotation_pass == "completion_chain" for item in specs) == 31
    assert sum(item.annotation_pass == "prefix_novelty" for item in specs) == 8
    assert len(specs) > 8
    descriptor = control_run.adapter.request_batch_descriptor(
        role="independent_a",
        request_specs=specs,
        config=control_run.adapter.load_adapter_config(
            expected_config_sha256=control_run.adapter.sha256_file(
                control_run.adapter.CONFIG_PATH
            )
        ),
    )
    assert descriptor["request_count"] == 39


def test_detail_round_is_derived_only_from_authenticated_chain_decisions():
    context = fake_context()
    manifest = control_manifest()
    specs = control_run.build_primary_initial_specs(
        role="independent_a",
        control_manifest=manifest,
        context=context,
        seed=17011,
    )
    chain_packet_ids = {
        item.packet_id_sha256
        for item in specs
        if item.annotation_pass == "completion_chain"
        and "chain" in item.schema["properties"]["decision"]["enum"]
    }
    selected = frozenset(sorted(chain_packet_ids)[:3])
    initial = batch_envelope(
        specs=specs, context=context, chain_packet_ids=selected
    )
    detail = control_run.build_primary_detail_specs(
        role="independent_a",
        control_manifest=manifest,
        context=context,
        seed=17011,
        initial_batch=initial,
    )
    assert len(detail) == 3
    assert {item.packet_id_sha256 for item in detail} == set(selected)
    assert all(
        set(item.lineage_bindings)
        == {
            "candidate_unit_bundle_sha256",
            "parent_decision_request_sha256",
            "parent_decision_result_sha256",
        }
        for item in detail
    )


def test_detail_specs_survive_canonical_json_round_trip_for_adapter_validation():
    context = fake_context()
    manifest = control_manifest()
    initial_specs = control_run.build_primary_initial_specs(
        role="independent_a",
        control_manifest=manifest,
        context=context,
        seed=17011,
    )
    selected = frozenset(
        [
            next(
                item.packet_id_sha256
                for item in initial_specs
                if item.annotation_pass == "completion_chain"
            )
        ]
    )
    initial = batch_envelope(
        specs=initial_specs,
        context=context,
        chain_packet_ids=selected,
    )
    detail = control_run.build_primary_detail_specs(
        role="independent_a",
        control_manifest=manifest,
        context=context,
        seed=17011,
        initial_batch=initial,
    )
    frozen = json.loads(
        control_run.canonical_json_bytes([asdict(item) for item in detail])
    )
    loaded = control_run._load_request_specs(frozen)

    assert list(loaded[0].schema["properties"]) == loaded[0].schema["required"]
    descriptor = control_run.adapter.request_batch_descriptor(
        role="independent_a",
        request_specs=loaded,
        config=control_run.adapter.load_adapter_config(
            expected_config_sha256=control_run.adapter.sha256_file(
                control_run.adapter.CONFIG_PATH
            )
        ),
    )
    assert descriptor["request_count"] == 1

    request = request_from_spec(loaded[0], context)
    restored_request = control_run._native_request(
        json.loads(control_run.canonical_json_bytes(asdict(request)))
    )
    assert restored_request.request_sha256 == request.request_sha256
    assert (
        list(restored_request.body["response_schema"]["properties"])
        == restored_request.body["response_schema"]["required"]
    )


def test_suite_init_writes_exact_inputs_nonce_and_live_trace(tmp_path: Path):
    generation = tmp_path / "generation"
    keys = tmp_path / "keys"
    generation.mkdir()
    keys.mkdir()
    result = control_run.init_suite(
        suite_id="test-development-suite-citrine-v3",
        generation_root=generation,
        key_root=keys,
        expected_executor_config_sha256=control_run.executor.sha256_file(
            control_run.executor.CONFIG_PATH
        ),
        expected_executor_source_sha256=control_run.executor.sha256_file(
            control_run.executor.SOURCE_PATH
        ),
        expected_controller_source_sha256=control_run.sha256_file(SOURCE),
    )
    receipt = control_run._validate_suite_init(
        control_run._read_exact_json(
            Path(result["suite_init_receipt_path"]),
            result["suite_init_receipt_file_sha256"],
            "suite init receipt",
        )
    )
    assert receipt["control_input_sha256"].startswith("cb2080")
    assert receipt["fixture_input_sha256"].startswith("81338a")
    assert len((generation / "suite.nonce").read_bytes()) == 32
    events = control_run.executor._parse_trace_journal(
        (generation / "read-trace.jsonl").read_bytes()
    )
    transitions = [
        event["artifact_id"] for event in events if event["event_type"] == "transition"
    ]
    assert transitions == ["freeze_complete", "nonce_precommitted"]
    assert all(event["stage_rank"] <= 1 for event in events)

    changed = copy.deepcopy(receipt)
    changed["controller_source_sha256"] = "0" * 64
    try:
        control_run._validate_suite_init(changed)
    except control_run.SealedControlRunError as error:
        assert str(error) == "controller source differs from suite precommit"
    else:
        raise AssertionError("changed controller source was accepted")
