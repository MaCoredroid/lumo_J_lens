#!/usr/bin/env python3
"""Tiny tests for the offline NF4 held-out evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
if torch is not None:
    SPEC = importlib.util.spec_from_file_location(
        "evaluate_jlens_nf4", SCRIPTS / "evaluate_jlens_nf4.py"
    )
    assert SPEC and SPEC.loader
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)
else:
    MODULE = None


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        offset = sum(text.encode("utf-8")) % 1000
        return {
            "input_ids": [offset + index for index in range(MODULE.PROMPT_TOKEN_COUNT)]
        }

    def decode(self, token_ids):
        return f"<{token_ids[0]}>"


def frozen_prompt_manifest(tokenizer):
    prompts = []
    for row_index in range(MODULE.MIN_PROMPTS):
        text = f"held out validation prompt {row_index}"
        token_ids = tokenizer(text)["input_ids"]
        prompts.append(
            {
                "row_index": row_index,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_count": MODULE.PROMPT_TOKEN_COUNT,
                "token_ids": token_ids,
            }
        )
    return {
        "schema_version": 1,
        "dataset": {
            "repo": MODULE.DATASET_REPO,
            "revision": MODULE.DATASET_REVISION,
            "config": MODULE.DATASET_CONFIG,
            "split": "validation",
        },
        "tokenizer": {
            "repo": MODULE.fitter.MODEL_ID,
            "revision": MODULE.fitter.MODEL_REVISION,
            "add_special_tokens": True,
            "force_bos_when_supported": True,
            "truncation": "right",
        },
        "selection": {
            "order": "dataset row order",
            "minimum_stripped_characters": 600,
            "required_token_count": MODULE.PROMPT_TOKEN_COUNT,
            "take": MODULE.MIN_PROMPTS,
        },
        "prompts": prompts,
    }


@unittest.skipIf(torch is None, "torch is installed in .venv-fit")
class EvaluatorHelperTest(unittest.TestCase):
    def test_production_defaults_cover_all_layers_and_frozen_positions(self):
        parser = MODULE.build_parser()
        self.assertEqual(parser.get_default("layers"), "all")
        self.assertEqual(parser.get_default("positions"), "16,32,64,96")
        self.assertEqual(
            MODULE.fitter.sha256_file(MODULE.PROMPTS_PATH), MODULE.PROMPTS_SHA256
        )

    def test_layer_and_position_parsing(self):
        self.assertEqual(MODULE.validate_layers([62, 0, 16]), [0, 16, 62])
        self.assertEqual(MODULE.resolve_positions([-2, 16], 128), [126, 16])
        with self.assertRaises(ValueError):
            MODULE.validate_layers([63])
        with self.assertRaises(ValueError):
            MODULE.resolve_positions([-1], 128)
        with self.assertRaises(ValueError):
            MODULE.resolve_positions([15], 128)

    def test_frozen_prompt_hash_and_token_ids(self):
        tokenizer = FakeTokenizer()
        payload = frozen_prompt_manifest(tokenizer)
        rendered = json.dumps(payload).encode()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            path.write_bytes(rendered)
            records, digest = MODULE.load_frozen_prompts(
                path,
                tokenizer,
                expected_sha256=hashlib.sha256(rendered).hexdigest(),
            )
            self.assertEqual(len(records), MODULE.MIN_PROMPTS)
            self.assertEqual(len(records[0]["token_ids"]), MODULE.PROMPT_TOKEN_COUNT)
            self.assertEqual(records[0]["dataset_split"], "validation")
            self.assertEqual(digest, hashlib.sha256(rendered).hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.load_frozen_prompts(
                    path, tokenizer, expected_sha256="0" * 64
                )

            payload["prompts"][0]["token_ids"][0] += 1
            altered = json.dumps(payload).encode()
            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "frozen token IDs"):
                MODULE.load_frozen_prompts(
                    path,
                    tokenizer,
                    expected_sha256=hashlib.sha256(altered).hexdigest(),
                )

    def test_post_block_capture_selects_positions(self):
        blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        positions = torch.tensor([0, 2])
        with MODULE.PostBlockCapture(blocks, [0, 1], positions) as capture:
            hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            hidden = blocks[0](hidden)
            blocks[1](hidden + 1)
        torch.testing.assert_close(capture.activations[0], hidden[0, [0, 2]])
        torch.testing.assert_close(capture.activations[1], hidden[0, [0, 2]] + 1)

    def test_jacobian_transport_is_transposed_and_norm_runs_once(self):
        class CountingNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                return value + 1

        norm = CountingNorm()
        head = nn.Linear(2, 2, bias=False, dtype=torch.float32)
        with torch.no_grad():
            head.weight.copy_(torch.eye(2))
        residual = torch.tensor([[1.0, 2.0]])
        jacobian = torch.tensor([[1.0, 0.0], [3.0, 1.0]])
        logits = MODULE.readout_logits(
            residual, norm, head, jacobian=jacobian
        )
        torch.testing.assert_close(logits, torch.tensor([[2.0, 6.0]]))
        self.assertEqual(norm.calls, 1)

    def test_targets_topk_and_rank(self):
        logits = torch.tensor([[0.0, 3.0, 2.0], [4.0, 1.0, 0.0]])
        targets, sources = MODULE.target_token_ids([7, 8, 9], [0, 1])
        self.assertEqual(targets, [8, 9])
        self.assertEqual(sources, ["teacher_forced_next_token"] * 2)
        records = MODULE.compact_topk(
            torch.tensor([[0.0, 3.0, 2.0]]), [2], FakeTokenizer(), top_k=2
        )
        self.assertEqual(records[0]["token_ids"], [1, 2])
        self.assertEqual(records[0]["target_rank"], 2)

    def test_pairwise_agreement_and_spearman_metrics(self):
        def record(token_ids, rank):
            return {
                "token_ids": token_ids,
                "target_token_id": 99,
                "target_rank": rank,
            }

        left = [
            record([1, 2, 3, 4, 5], 1),
            record([8, 9, 10, 11, 12], 3),
        ]
        right = [
            record([1, 2, 3, 6, 7], 2),
            record([9, 8, 10, 11, 12], 4),
        ]
        metrics = MODULE.compare_method_records(left, right)
        self.assertEqual(metrics["top1_agreement_count"], 1)
        self.assertEqual(metrics["top5_exact_set_agreement_count"], 1)
        self.assertEqual(metrics["top5_overlap_count"], 8)
        self.assertEqual(metrics["top5_overlap_mean_fraction"], 0.8)
        self.assertEqual(metrics["spearman_target_rank"]["coefficient"], 1.0)
        inverse = MODULE.spearman_target_rank([1, 2, 3], [3, 2, 1])
        self.assertEqual(inverse["coefficient"], -1.0)

    def test_summary_wires_local_public_and_logit_public_comparisons(self):
        def record(token_ids, rank):
            return {
                "token_ids": token_ids,
                "target_token_id": 99,
                "target_rank": rank,
            }

        experiments = []
        for index in range(2):
            public = record([1, 2, 3, 4, 5], index + 1)
            local = record([1, 2, 3, 6, 7], index + 2)
            logit = record([8, 2, 3, 4, 5], index + 3)
            experiments.append(
                {
                    "targets": [{"token_position": 16}],
                    "layers": [
                        {
                            "layer": 0,
                            "positions": [
                                {
                                    "token_position": 16,
                                    "vanilla_logit_lens": logit,
                                    "local_jacobian_lens": local,
                                    "public_jacobian_lens": public,
                                }
                            ],
                        }
                    ],
                }
            )
        summary = MODULE.summarize(experiments, [0])[0]
        self.assertEqual(
            set(summary["comparisons"]), {"local_vs_public", "logit_vs_public"}
        )
        self.assertEqual(
            summary["comparisons"]["local_vs_public"]["top1_agreement_rate"],
            1.0,
        )

    def test_evaluation_nf4_must_match_fit_aggregate(self):
        digest = "a" * 64
        provenance = {
            "model": {"quantized_weights": {"aggregate_sha256": digest}}
        }
        metadata = {"quantized_weights": {"aggregate_sha256": digest}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.json"
            path.write_text(json.dumps(provenance), encoding="utf-8")
            self.assertEqual(
                MODULE.require_matching_nf4_aggregate(path, metadata), digest
            )
            metadata["quantized_weights"]["aggregate_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.require_matching_nf4_aggregate(path, metadata)

    def test_eval_head_must_be_distinct_and_untied(self):
        def model(tied=False):
            value = types.SimpleNamespace()
            value.config = types.SimpleNamespace(tie_word_embeddings=tied)
            value.model = types.SimpleNamespace(embed_tokens=nn.Embedding(3, 2))
            value.lm_head = nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
            for parameter in value.lm_head.parameters():
                parameter.requires_grad_(False)
            return value

        valid = model()
        metadata = MODULE.prepare_eval_lm_head(
            valid, torch.device("cpu"), require_cuda_embedding=False
        )
        self.assertTrue(metadata["untied"])
        with self.assertRaisesRegex(RuntimeError, "untied"):
            MODULE.prepare_eval_lm_head(
                model(tied=True), torch.device("cpu"), require_cuda_embedding=False
            )


if __name__ == "__main__":
    unittest.main()
