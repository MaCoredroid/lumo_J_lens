from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts"
    / "swe_task_state_v4_epistemic_chain_sealed_adjudicator_run_v3.py"
)
SPEC = importlib.util.spec_from_file_location("sealed_adjudicator_run_v3_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
adjudicator_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adjudicator_run
SPEC.loader.exec_module(adjudicator_run)
control_run = adjudicator_run.control_run
import swe_task_state_v4_epistemic_chain_control_key_draft_v3 as control_key  # noqa: E402
import swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3 as scorer  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        digest = control_run.sha256_value(list(messages))
        return [11, int(digest[:8], 16), len(messages), 12]


def context_for(role: str):
    model = {
        "base_model_lineage": f"test-{role}",
        "repo_id": f"test/{role}",
        "revision": "test-revision",
        "snapshot_tree_sha256": control_run.sha256_value(f"snapshot-{role}"),
        "quantization": "test",
        "dtype": "bfloat16",
    }
    tokenizer = {
        "repo_id": f"test/{role}",
        "revision": "test-revision",
        "snapshot_tree_sha256": model["snapshot_tree_sha256"],
        "tokenizer_mode": "auto",
        "tokenizer_class": "tests.FakeTokenizer",
        "vocab_identity_sha256": control_run.sha256_value(f"vocab-{role}"),
    }
    return adjudicator_run.runner.authenticate_native_generation_context(
        tokenizer=FakeTokenizer(),
        model_identity=model,
        expected_model_identity_sha256=control_run.sha256_value(model),
        tokenizer_identity=tokenizer,
        expected_tokenizer_identity_sha256=control_run.sha256_value(tokenizer),
        chat_template_kwargs={},
    )


def fake_load_context(*, authenticated_adapter, role):
    del authenticated_adapter
    return (
        context_for(role),
        {"runtime_identity_sha256": control_run.sha256_value("runtime")},
        {"environment_identity_sha256": control_run.sha256_value("environment")},
        {"package_bundle_sha256": control_run.sha256_value("packages")},
        {"inventory_sha256": control_run.sha256_value(f"inventory-{role}")},
    )


def artifact_from_specs(*, specs, role: str, round_name: str, text_for):
    context = context_for(role)
    requests = [
        adjudicator_run.runner.build_native_generation_request(
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
        for spec in specs
    ]
    results = []
    for request in requests:
        ids = request.body["submitted_prompt_token_ids"]
        results.append(
            adjudicator_run.runner.build_native_generation_result(
                request=request,
                text=text_for(request),
                submitted_prompt_token_ids=ids,
                engine_prompt_token_ids=ids,
                output_token_ids=[91],
                finish_reason="stop",
            )
        )
    preflight = {"role": role, "round": round_name, "kind": "preflight"}
    runtime = {"role": role, "round": round_name, "kind": "runtime"}
    artifact = {
        "schema_version": 1,
        "interface_version": 3,
        "kind": control_run.BATCH_ARTIFACT_KIND,
        "status": "authenticated_native_batch_complete",
        "suite_id": "synthetic-development-suite-citrine-v3",
        "role": role,
        "round": round_name,
        "freeze_manifest_file_sha256": control_run.sha256_value(
            f"freeze-{role}-{round_name}"
        ),
        "adapter_launch_file_sha256": control_run.sha256_value(
            f"adapter-{role}-{round_name}"
        ),
        "outer_launch_file_sha256": control_run.sha256_value(
            f"outer-{role}-{round_name}"
        ),
        "request_count": len(requests),
        "requests": [asdict(item) for item in requests],
        "results": [asdict(item) for item in results],
        "preflight_receipt": {
            "body": preflight,
            "receipt_sha256": control_run.sha256_value(preflight),
        },
        "runtime_receipt": {
            "body": runtime,
            "receipt_sha256": control_run.sha256_value(runtime),
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
    envelope = {
        "artifact": artifact,
        "artifact_sha256": control_run.sha256_value(artifact),
    }
    control_run.validate_batch_artifact(
        envelope, expected_role=role, expected_round=round_name
    )
    return envelope


def write_json(path: Path, value) -> str:
    return control_run._exclusive_write_json(path, value, str(path))


def test_synthetic_full_protocol_closes_and_authenticates_public_bundle(
    tmp_path: Path, monkeypatch
):
    # Other V3 tests intentionally load frozen sources under private module
    # aliases. Normalize those module globals so dataclass identity does not
    # depend on pytest collection order; production imports use one canonical
    # module name and do not need this test-only normalization.
    monkeypatch.setattr(adjudicator_run.fixture, "production", adjudicator_run.runner)
    monkeypatch.setattr(adjudicator_run.fixture_key, "runner", adjudicator_run.runner)
    monkeypatch.setattr(adjudicator_run.fixture_key, "fixture", adjudicator_run.fixture)
    monkeypatch.setattr(control_run, "_load_context", fake_load_context)
    # The fixture protocol intentionally invalidates an imbalanced nonce with no
    # retry. Keep this end-to-end test deterministic with one preselected test
    # nonce whose first and only presentation is the required 4/4/4 split.
    monkeypatch.setattr(
        control_run.executor.secrets,
        "token_bytes",
        lambda length: (3).to_bytes(length, "big"),
    )
    generation = tmp_path / "generation"
    generation.mkdir()
    # The traced, fixed-source fixture key lives under this root. The temporary
    # generation tree is physically disjoint from the repository.
    key_root = ROOT
    executor_config_sha = control_run.executor.sha256_file(
        control_run.executor.CONFIG_PATH
    )
    executor_source_sha = control_run.executor.sha256_file(
        control_run.executor.SOURCE_PATH
    )
    adapter_config_sha = control_run.adapter.sha256_file(control_run.adapter.CONFIG_PATH)
    adapter_source_sha = control_run.adapter.sha256_file(control_run.adapter.SOURCE_PATH)
    controller_source_sha = control_run.sha256_file(Path(control_run.__file__).resolve())
    adjudicator_source_sha = control_run.sha256_file(SOURCE)
    init = control_run.init_suite(
        suite_id="synthetic-development-suite-citrine-v3",
        generation_root=generation,
        key_root=key_root,
        expected_executor_config_sha256=executor_config_sha,
        expected_executor_source_sha256=executor_source_sha,
        expected_controller_source_sha256=controller_source_sha,
    )
    suite_path = Path(init["suite_init_receipt_path"])
    suite_sha = init["suite_init_receipt_file_sha256"]
    journal_sha = init["trace_journal_file_sha256"]
    primary_results = {}
    for role in ("independent_a", "independent_b"):
        round_dir = generation / f"{role}-initial"
        frozen = control_run.freeze_primary_round(
            role=role,
            round_name="initial",
            suite_init_path=suite_path,
            expected_suite_init_file_sha256=suite_sha,
            expected_executor_config_sha256=executor_config_sha,
            expected_executor_source_sha256=executor_source_sha,
            expected_adapter_config_sha256=adapter_config_sha,
            expected_adapter_source_sha256=adapter_source_sha,
            expected_controller_source_sha256=controller_source_sha,
            output_directory=round_dir,
        )
        freeze = control_run._read_exact_json(
            Path(frozen["freeze_manifest_path"]),
            frozen["freeze_manifest_file_sha256"],
            "primary freeze",
        )
        specs = control_run._load_request_specs(
            control_run._read_exact_json(
                Path(freeze["request_specs_path"]),
                freeze["request_specs_file_sha256"],
                "primary specs",
            )
        )
        chain_packet_id = next(
            item.packet_id_sha256
            for item in specs
            if item.annotation_pass == "completion_chain"
        )
        envelope = artifact_from_specs(
            specs=specs,
            role=role,
            round_name="initial",
            text_for=lambda request: (
                '{"decision":"novel"}'
                if request.body["annotation_pass"] == "prefix_novelty"
                else (
                    '{"decision":"chain"}'
                    if request.body["packet_id_sha256"] == chain_packet_id
                    else '{"decision":"no_chain"}'
                )
            ),
        )
        batch_path = round_dir / "synthetic-batch.json"
        batch_sha = write_json(batch_path, envelope)
        suite = control_run._validate_suite_init(
            control_run._read_exact_json(suite_path, suite_sha, "suite")
        )
        journal_sha = control_run.executor.read_exact_and_append_trace(
            journal_path=Path(suite["trace_journal_path"]),
            expected_journal_file_sha256=journal_sha,
            artifact_path=Path(suite["control_input_path"]),
            expected_artifact_sha256=suite["control_input_sha256"],
            stage="run_primary",
            artifact_class="control_input",
            artifact_id=(
                "control_manifest_regenerated_for_a"
                if role == "independent_a"
                else "control_manifest_regenerated_for_b"
            ),
        )["journal_file_sha256"]
        detail_dir = generation / f"{role}-detail"
        detail = control_run.freeze_primary_round(
            role=role,
            round_name="detail",
            suite_init_path=suite_path,
            expected_suite_init_file_sha256=suite_sha,
            expected_executor_config_sha256=executor_config_sha,
            expected_executor_source_sha256=executor_source_sha,
            expected_adapter_config_sha256=adapter_config_sha,
            expected_adapter_source_sha256=adapter_source_sha,
            expected_controller_source_sha256=controller_source_sha,
            output_directory=detail_dir,
            initial_batch_path=batch_path,
            expected_initial_batch_file_sha256=batch_sha,
        )
        assert detail["request_count"] == 1
        detail_freeze = control_run._read_exact_json(
            Path(detail["freeze_manifest_path"]),
            detail["freeze_manifest_file_sha256"],
            "detail freeze",
        )
        detail_specs = control_run._load_request_specs(
            control_run._read_exact_json(
                Path(detail_freeze["request_specs_path"]),
                detail_freeze["request_specs_file_sha256"],
                "detail specs",
            )
        )

        def detail_text(request):
            properties = request.body["response_schema"]["properties"]
            unit_ids = properties["evidence_unit_id"]["enum"]
            return json.dumps(
                {
                    "evidence_unit_id": unit_ids[0],
                    "hypothesis_unit_id": unit_ids[1],
                    "action_unit_id": unit_ids[2],
                    "evidence_kind": properties["evidence_kind"]["enum"][0],
                    "belief_edge": properties["belief_edge"]["enum"][0],
                    "hypothesis_domain": properties["hypothesis_domain"]["enum"][0],
                    "action_intent": properties["action_intent"]["enum"][0],
                },
                sort_keys=True,
                separators=(",", ":"),
            )

        detail_envelope = artifact_from_specs(
            specs=detail_specs,
            role=role,
            round_name="detail",
            text_for=detail_text,
        )
        detail_batch_path = detail_dir / "synthetic-batch.json"
        detail_batch_sha = write_json(detail_batch_path, detail_envelope)
        finalized = control_run.finalize_primary(
            role=role,
            suite_init_path=suite_path,
            expected_suite_init_file_sha256=suite_sha,
            initial_batch_path=batch_path,
            expected_initial_batch_file_sha256=batch_sha,
            detail_freeze_manifest_path=Path(detail["freeze_manifest_path"]),
            expected_detail_freeze_manifest_file_sha256=detail[
                "freeze_manifest_file_sha256"
            ],
            detail_batch_path=detail_batch_path,
            expected_detail_batch_file_sha256=detail_batch_sha,
            expected_adapter_config_sha256=adapter_config_sha,
            expected_adapter_source_sha256=adapter_source_sha,
            expected_trace_journal_file_sha256=journal_sha,
        )
        journal_sha = finalized["trace_journal_file_sha256"]
        primary_results[role] = finalized
    dual = adjudicator_run.lock_primaries(
        suite_init_path=suite_path,
        expected_suite_init_file_sha256=suite_sha,
        primary_a_path=Path(primary_results["independent_a"]["primary_receipt_path"]),
        expected_primary_a_file_sha256=primary_results["independent_a"][
            "primary_receipt_file_sha256"
        ],
        primary_b_path=Path(primary_results["independent_b"]["primary_receipt_path"]),
        expected_primary_b_file_sha256=primary_results["independent_b"][
            "primary_receipt_file_sha256"
        ],
        expected_trace_journal_file_sha256=journal_sha,
        expected_adjudicator_source_sha256=adjudicator_source_sha,
    )
    journal_sha = dual["trace_journal_file_sha256"]
    verdict_dir = generation / "adjudicator-verdict"
    verdict_freeze = adjudicator_run.prepare_and_freeze_verdict(
        suite_init_path=suite_path,
        expected_suite_init_file_sha256=suite_sha,
        dual_primary_lock_path=Path(dual["dual_primary_lock_path"]),
        expected_dual_primary_lock_file_sha256=dual[
            "dual_primary_lock_file_sha256"
        ],
        expected_executor_config_sha256=executor_config_sha,
        expected_executor_source_sha256=executor_source_sha,
        expected_adapter_config_sha256=adapter_config_sha,
        expected_adapter_source_sha256=adapter_source_sha,
        expected_adjudicator_source_sha256=adjudicator_source_sha,
        expected_trace_journal_file_sha256=journal_sha,
        output_directory=verdict_dir,
        fixture_key_output_path=generation / "synthetic-fixture-key.json",
    )
    journal_sha = verdict_freeze["trace_journal_file_sha256"]
    verdict_freeze_value = control_run._read_exact_json(
        Path(verdict_freeze["freeze_manifest_path"]),
        verdict_freeze["freeze_manifest_file_sha256"],
        "verdict freeze",
    )
    verdict_specs = control_run._load_request_specs(
        control_run._read_exact_json(
            Path(verdict_freeze_value["request_specs_path"]),
            verdict_freeze_value["request_specs_file_sha256"],
            "verdict specs",
        )
    )
    neither_packet_id = verdict_specs[0].packet_id_sha256
    verdict_envelope = artifact_from_specs(
        specs=verdict_specs,
        role="adjudicator",
        round_name="verdict",
        text_for=lambda request: (
            '{"verdict":"neither"}'
            if request.body["packet_id_sha256"] == neither_packet_id
            else '{"verdict":"candidate_1"}'
        ),
    )
    verdict_batch_path = verdict_dir / "synthetic-batch.json"
    verdict_batch_sha = write_json(verdict_batch_path, verdict_envelope)
    preparation_path = Path(verdict_freeze_value["preparation_path"])
    preparation_sha = verdict_freeze_value["preparation_file_sha256"]
    repair = adjudicator_run.freeze_followup_round(
        round_name="repair",
        suite_init_path=suite_path,
        expected_suite_init_file_sha256=suite_sha,
        preparation_path=preparation_path,
        expected_preparation_file_sha256=preparation_sha,
        verdict_batch_path=verdict_batch_path,
        expected_verdict_batch_file_sha256=verdict_batch_sha,
        expected_executor_config_sha256=executor_config_sha,
        expected_executor_source_sha256=executor_source_sha,
        expected_adapter_config_sha256=adapter_config_sha,
        expected_adapter_source_sha256=adapter_source_sha,
        expected_adjudicator_source_sha256=adjudicator_source_sha,
        output_directory=generation / "adjudicator-repair",
    )
    assert repair["request_count"] == 1
    repair_freeze = control_run._read_exact_json(
        Path(repair["freeze_manifest_path"]),
        repair["freeze_manifest_file_sha256"],
        "repair freeze",
    )
    repair_specs = control_run._load_request_specs(
        control_run._read_exact_json(
            Path(repair_freeze["request_specs_path"]),
            repair_freeze["request_specs_file_sha256"],
            "repair specs",
        )
    )
    repair_envelope = artifact_from_specs(
        specs=repair_specs,
        role="adjudicator",
        round_name="repair",
        text_for=lambda _request: '{"decision":"chain"}',
    )
    repair_batch_path = generation / "adjudicator-repair" / "synthetic-batch.json"
    repair_batch_sha = write_json(repair_batch_path, repair_envelope)
    detail = adjudicator_run.freeze_followup_round(
        round_name="detail",
        suite_init_path=suite_path,
        expected_suite_init_file_sha256=suite_sha,
        preparation_path=preparation_path,
        expected_preparation_file_sha256=preparation_sha,
        verdict_batch_path=verdict_batch_path,
        expected_verdict_batch_file_sha256=verdict_batch_sha,
        repair_batch_path=repair_batch_path,
        expected_repair_batch_file_sha256=repair_batch_sha,
        expected_executor_config_sha256=executor_config_sha,
        expected_executor_source_sha256=executor_source_sha,
        expected_adapter_config_sha256=adapter_config_sha,
        expected_adapter_source_sha256=adapter_source_sha,
        expected_adjudicator_source_sha256=adjudicator_source_sha,
        output_directory=generation / "adjudicator-detail",
    )
    assert detail["request_count"] == 1
    adjudicator_detail_freeze = control_run._read_exact_json(
        Path(detail["freeze_manifest_path"]),
        detail["freeze_manifest_file_sha256"],
        "adjudicator detail freeze",
    )
    adjudicator_detail_specs = control_run._load_request_specs(
        control_run._read_exact_json(
            Path(adjudicator_detail_freeze["request_specs_path"]),
            adjudicator_detail_freeze["request_specs_file_sha256"],
            "adjudicator detail specs",
        )
    )
    adjudicator_detail_envelope = artifact_from_specs(
        specs=adjudicator_detail_specs,
        role="adjudicator",
        round_name="detail",
        text_for=detail_text,
    )
    adjudicator_detail_batch_path = (
        generation / "adjudicator-detail" / "synthetic-batch.json"
    )
    adjudicator_detail_batch_sha = write_json(
        adjudicator_detail_batch_path, adjudicator_detail_envelope
    )
    finalized_j = adjudicator_run.finalize_adjudicator(
        suite_init_path=suite_path,
        expected_suite_init_file_sha256=suite_sha,
        preparation_path=preparation_path,
        expected_preparation_file_sha256=preparation_sha,
        verdict_batch_path=verdict_batch_path,
        expected_verdict_batch_file_sha256=verdict_batch_sha,
        repair_freeze_path=Path(repair["freeze_manifest_path"]),
        expected_repair_freeze_file_sha256=repair["freeze_manifest_file_sha256"],
        repair_batch_path=repair_batch_path,
        expected_repair_batch_file_sha256=repair_batch_sha,
        detail_freeze_path=Path(detail["freeze_manifest_path"]),
        expected_detail_freeze_file_sha256=detail["freeze_manifest_file_sha256"],
        detail_batch_path=adjudicator_detail_batch_path,
        expected_detail_batch_file_sha256=adjudicator_detail_batch_sha,
        expected_adapter_config_sha256=adapter_config_sha,
        expected_adapter_source_sha256=adapter_source_sha,
        expected_trace_journal_file_sha256=journal_sha,
    )
    locked = adjudicator_run.lock_all(
        suite_init_path=suite_path,
        expected_suite_init_file_sha256=suite_sha,
        dual_primary_lock_path=Path(dual["dual_primary_lock_path"]),
        expected_dual_primary_lock_file_sha256=dual[
            "dual_primary_lock_file_sha256"
        ],
        adjudicator_receipt_path=Path(finalized_j["adjudicator_receipt_path"]),
        expected_adjudicator_receipt_file_sha256=finalized_j[
            "adjudicator_receipt_file_sha256"
        ],
        expected_executor_config_sha256=executor_config_sha,
        expected_executor_source_sha256=executor_source_sha,
        expected_adapter_config_sha256=adapter_config_sha,
        expected_adapter_source_sha256=adapter_source_sha,
        expected_trace_journal_file_sha256=finalized_j[
            "trace_journal_file_sha256"
        ],
    )
    assert locked["claims"]["sealed_development_generation_bundle_complete"] is True
    assert locked["claims"]["scoring_performed"] is False
    assert locked["event_count"] >= 20
    suite = control_run._validate_suite_init(
        control_run._read_exact_json(suite_path, suite_sha, "suite")
    )
    main_key = control_key.materialize_control_key_draft(
        input_manifest=control_run._read_exact_json(
            Path(suite["control_input_path"]),
            suite["control_input_sha256"],
            "control input",
        ),
        independently_supplied_manifest_sha256=suite["control_input_sha256"],
    )
    main_key_path = generation / "synthetic-main-key.json"
    main_key_file_sha = write_json(main_key_path, main_key)
    preparation = adjudicator_run._validate_preparation(
        control_run._read_exact_json(
            preparation_path, preparation_sha, "adjudicator preparation"
        ),
        suite_id=suite["suite_id"],
    )
    score_path = generation / "synthetic-score-receipt.json"
    scorer_config_path = (
        ROOT
        / "configs"
        / "swe_task_state_v4_epistemic_chain_sealed_control_scorer_v3.json"
    )
    score = scorer.score_finalized_executor_bundle(
        config_path=scorer_config_path,
        expected_config_sha256=scorer.sha256_bytes(scorer_config_path.read_bytes()),
        expected_scorer_source_sha256=scorer.sha256_bytes(
            Path(scorer.__file__).resolve().read_bytes()
        ),
        expected_executor_config_sha256=executor_config_sha,
        expected_executor_source_sha256=executor_source_sha,
        read_trace_path=Path(locked["read_trace_path"]),
        expected_read_trace_file_sha256=locked["read_trace_file_sha256"],
        expected_read_trace_envelope_sha256=locked[
            "read_trace_envelope_sha256"
        ],
        public_bundle_path=Path(locked["public_bundle_path"]),
        expected_public_bundle_file_sha256=locked["public_bundle_file_sha256"],
        expected_public_bundle_envelope_sha256=locked[
            "public_bundle_envelope_sha256"
        ],
        main_control_key_path=main_key_path,
        expected_main_control_key_file_sha256=main_key_file_sha,
        expected_main_control_key_manifest_sha256=main_key["manifest_sha256"],
        fixture_adjudication_key_path=Path(preparation["fixture_key_path"]),
        expected_fixture_adjudication_key_file_sha256=preparation[
            "fixture_key_file_sha256"
        ],
        expected_fixture_adjudication_key_manifest_sha256=preparation[
            "fixture_key_manifest_sha256"
        ],
        receipt_path=score_path,
    )
    assert score["receipt"]["status"] == (
        "development_controls_scored_not_sealed_not_gate_evidence"
    )
    assert all(value is False for value in score["receipt"]["claims"].values())
