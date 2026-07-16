#!/usr/bin/env python3
"""Run Qwen Code headlessly on SWE-bench Verified against a local vLLM server.

This is the standalone, path-neutral extraction of the runner certified in the
Qwen3.6-27B NVFP4 + MTP teacher session on 2026-07-08. Qwen Code itself runs on
the host from an isolated directory containing only AGENTS.md. Its sole declared
tool, run_shell_command, is redirected into the official per-instance container.
The final artifact is a normal SWE-bench predictions JSONL file.

Per-instance protocol:
  1. Seed /testbed from the pinned official image and bind it into a long-lived
     container; write AGENTS.md carrying the problem_statement.
  2. Run Qwen Code headlessly from an isolated host CWD, with its shell shim
     targeting the container and qwen_code_proxy.py targeting vLLM.
  3. Diff the in-container workspace vs base_commit -> patch.diff.
  4. Emit eval metadata and predictions.jsonl. The official scorer is invoked
     separately by scripts/score_verified.sh.
  5. Write per-task artifacts under <out>/<dataset>/per_task/<instance_id>/.
  6. Aggregate predictions.jsonl + campaign_summary.json.

Qwen Code may exit nonzero after leaving a useful patch (for example, a loop
detector or wall-time exit). The CLI exit code is diagnostic only. The official
SWE-bench harness result is the verdict.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from check_endpoint import validate_models_payload
from resolve_swe_image import resolve_image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "swe_verified"
DEFAULT_REPO_CACHE = REPO_ROOT / ".cache" / "swe_bench_repos"
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
DEFAULT_QWEN_BIN = REPO_ROOT / "node_modules" / ".bin" / "qwen"
DEFAULT_PROXY_SCRIPT = REPO_ROOT / "scripts" / "qwen_code_proxy.py"

# Endpoint defaults follow the Stage-A certified serving ports.
DEFAULT_ENDPOINT = "http://127.0.0.1:9952/v1"
DEFAULT_MODEL = "qwen3.6-27b-nvfp4"
# qwen-code version bumped 0.19.2 -> 0.19.4 to match the flywheel's RESOLVED agent
# (run_swe_bench_q36_a.py DEFAULT_MODEL_NAME_TAG / qwen-code-runner.Dockerfile pin
# / prepare_qwen_agent_bundle.sh @0.19.4). Keep the ::qwen-code-<ver>:: segment in
# sync with node_modules/@qwen-code/qwen-code so predictions.jsonl records the agent.
DEFAULT_MODEL_NAME_TAG = "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4"

# Per-attempt agent wall (subprocess timeout). 0 => rely on the qwen CLI's own
# --max-wall-time / --max-session-turns budgets (like the flywheel's codex idle
# timeout backstop). Set >0 for a hard harness wall.
DEFAULT_AGENT_WALL_S = 0
# Flywheel qwen-code profile (run_swe_bench_q36_a.py, user 2026-07-07): NO turn cap —
# `--max-session-turns 100000` is effectively unlimited so a task runs to its natural
# submit/give-up. Backstop = QWEN_STREAM_IDLE_TIMEOUT_MS (+ optional harness/qwen wall).
DEFAULT_MAX_SESSION_TURNS = 100000
# Flywheel default: NO `--max-wall-time` (empty => flag omitted; rely on the stream-idle
# timeout below). Set --qwen-max-wall "1800s" for a hard qwen run-level budget (exit 55).
DEFAULT_QWEN_MAX_WALL = ""
DEFAULT_QWEN_MAX_OUTPUT_TOKENS = 32768   # flywheel R1 context-budget fix
# Flywheel FR13 §59/§79 belt: raise qwen-code's stream-idle abort (built-in default
# 120000ms) so a transient mid-stream upstream flake or a pre-first-byte queue wait is
# not converted into a fatal patch-less give-up. qwen-code 0.19.4 reads
# QWEN_STREAM_IDLE_TIMEOUT_MS (precedence: config field > env > default). 240000 is the
# flywheel QWEN_CODE_TEMPLATE value; their in-image path uses 600000 for B>=4 queueing.
DEFAULT_QWEN_STREAM_IDLE_TIMEOUT_MS = 240000
DEFAULT_EVAL_TIMEOUT_S = 30 * 60

# Proxy adapter: inject the certified thinking envelope, clamp max_tokens, and
# enforce the one-tool run_shell_command schema before forwarding to vLLM. The
# qwen3_xml parser runs server-side; natural tool_choice still permits a final
# free-text turn.
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 30021
DEFAULT_PROXY_MAX_TOKENS = 8192   # thinking-teacher headroom (CONFIG_DELTAS.md D7);
DEFAULT_DATASET_REVISION = os.environ.get(
    "SWE_DATASET_REVISION", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
)
#   the 27B Regime-T <think> trace + patch needs >2048/turn (gate-2 ran 8192 direct,
#   0.9844 verbatim). Non-thinking 9B rarely emits >2048/turn, so this is a safe cap
#   for both teachers. Was 2048 (inherited 9B toy-smoke bump over the 512 default).

# ---------------------------------------------------------------------------
# Operator prompts (ported from the flywheel and adapted to the shell-only
# container boundary).
# ---------------------------------------------------------------------------
DEFAULT_AGENT_PROMPT = (
    "Read the task prompt at ./AGENTS.md and complete it in this workspace. "
    "Use run_shell_command for every inspection, edit, and test; it already runs "
    "inside the official task container at /testbed, so use relative paths and do "
    "not cd to the host workspace path. Edit the source files directly to implement "
    "the fix. Do not write a diff file -- modify the files in place so that running "
    "the project's tests passes the tests described in the prompt. Do NOT modify "
    "any test files."
)
RETRY_PROMPT_EMPTY = (
    "Your previous attempt finished WITHOUT leaving any code change in the working "
    "tree. Re-read ./AGENTS.md, inspect the relevant source files, and EDIT them "
    "now to implement the fix. Do not stop until you have made a concrete source "
    "edit. Do not waste time on environment setup or pip/conda installs -- the "
    "grader uses its own environment."
)
RETRY_PROMPT_SETUP_LOOP = (
    "Your previous attempt repeatedly hit the same failing command (likely an "
    "environment/install/build step) and never edited the source. STOP trying that "
    "approach entirely. The grader builds its own environment, so you do NOT need "
    "the project to install or import. Read ./AGENTS.md and the relevant source "
    "files, and directly EDIT the source to implement the fix."
)


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*__[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _safe_task_dir(root: Path, instance_id: str) -> Path:
    if not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError(f"unsafe SWE instance_id: {instance_id!r}")
    resolved_root = root.resolve()
    task_dir = (resolved_root / instance_id).resolve()
    if task_dir.parent != resolved_root:
        raise ValueError(f"instance_id escapes output root: {instance_id!r}")
    return task_dir


def _validated_commit(value: object) -> str:
    commit = str(value)
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError(f"unsafe base_commit: {commit!r}")
    return commit


def _empty_patch_retries(args) -> int:
    """Re-drive count for the empty-patch (tool-call-free-terminal) mitigation.

    Precedence: --empty-patch-retries (if given) > env SWE_EMPTY_PATCH_RETRIES >
    0. Clamped to >= 0. Default 0 keeps a plain single attempt (byte-identical to
    the pre-envelope behaviour); the reference-envelope run sets >= 1."""
    cli = getattr(args, "empty_patch_retries", None)
    if cli is not None:
        return max(0, int(cli))
    return max(0, int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")))


# ---------------------------------------------------------------------------
# Dataset / subset loading (ported verbatim from the flywheel).
# ---------------------------------------------------------------------------
def _load_subset(subset_json: Path) -> tuple[str, list[str]]:
    payload = json.loads(subset_json.read_text())
    return payload["dataset_name"], list(payload["instance_ids"])


def _load_dataset(
    dataset_name: str, split: str = "test", revision: str | None = None
) -> dict[str, dict]:
    # Local .json / .jsonl (e.g. the SWE-Gym probe pool dumped to a file) — no HF,
    # arbitrary split. Keeps the container/patch path identical; only the record
    # source differs.
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        p = Path(dataset_name)
        if dataset_name.endswith(".jsonl"):
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        else:
            recs = json.loads(p.read_text())
        return {ex["instance_id"]: dict(ex) for ex in recs}
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    from datasets import load_dataset

    kwargs = {"split": split}
    if revision and dataset_name == "princeton-nlp/SWE-bench_Verified":
        kwargs["revision"] = revision
    ds = load_dataset(dataset_name, **kwargs)
    return {ex["instance_id"]: dict(ex) for ex in ds}


# ---------------------------------------------------------------------------
# Workspace hydrate / teardown (ported verbatim from the flywheel).
# ---------------------------------------------------------------------------
def _repo_clone_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _ensure_repo_cache(repo: str, cache_root: Path) -> Path:
    safe = repo.replace("/", "__")
    cache_path = cache_root / safe
    if not cache_path.is_dir():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _repo_clone_url(repo), str(cache_path)],
            check=True,
        )
    return cache_path


def _fetch_commit(cache_path: Path, base_commit: str) -> None:
    rc = subprocess.run(
        ["git", "-C", str(cache_path), "cat-file", "-e", base_commit]
    ).returncode
    if rc != 0:
        subprocess.run(
            ["git", "-C", str(cache_path), "fetch", "origin", base_commit],
            check=False,
        )


def _hydrate_workspace(*, cache_path: Path, base_commit: str, workspace_path: Path) -> None:
    if workspace_path.exists():
        _remove_workspace(cache_path, workspace_path)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    abs_workspace = workspace_path.resolve()
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "add", "--detach",
         str(abs_workspace), base_commit],
        check=True,
    )


def _remove_workspace(cache_path: Path, workspace_path: Path) -> None:
    abs_workspace = workspace_path.resolve() if workspace_path.exists() else workspace_path
    if not abs_workspace.exists():
        return
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "remove", "--force", str(abs_workspace)],
        check=False,
    )
    if abs_workspace.exists():
        shutil.rmtree(abs_workspace, ignore_errors=True)


def _write_agents_md(workspace: Path, instance: dict) -> None:
    """AGENTS.md drop — byte-for-byte the flywheel convention so the task prompt
    the agent sees is identical to the flywheel's Codex campaigns."""
    body: list[str] = []
    body.append(f"# SWE-Bench task: {instance['instance_id']}")
    body.append("")
    body.append(f"**Repo:** `{instance['repo']}`  ")
    body.append(f"**Base commit:** `{instance['base_commit']}`  ")
    if instance.get("version"):
        body.append(f"**Version:** `{instance['version']}`  ")
    body.append("")
    body.append("## Problem statement")
    body.append("")
    body.append(instance.get("problem_statement") or "(empty problem statement)")
    body.append("")
    body.append("## Required behavior")
    body.append("")
    body.append(
        "Implement the fix described in the problem statement by editing the "
        "source files in this workspace. Do NOT modify any test files. The "
        "hidden grader will apply its own test patch and run the test suite; "
        "your code must make those tests pass without breaking existing ones."
    )
    body.append("")
    body.append("## How to work (important)")
    body.append("")
    body.append(
        "- Reason carefully and thoroughly before each tool call. First inspect "
        "the relevant source files to confirm your understanding of the bug, "
        "then make the minimal correct edit.\n"
        "- Do NOT spend your time trying to `pip install` or build/conda the "
        "project -- the grader runs in its own prepared environment. If an "
        "install/build command fails, do not retry it; just edit the source.\n"
        "- You MUST finish by leaving an actual code change in the working tree. "
        "Do not stop until you have edited the source files to implement the fix."
    )
    body.append("")
    (workspace / "AGENTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def _extract_patch(workspace: Path, base_commit: str) -> str:
    """Ported verbatim: tracked-file diff vs base_commit (binary-safe)."""
    proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-color", "--binary", base_commit],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


# ---------------------------------------------------------------------------
# CONTAINER RUNTIME (RUNTIME_ALIGNMENT_DIRECTIVE, 2026-07-05).
#
# Official SWE-bench Verified hands the agent the per-instance RUNTIME: the repo
# checked out at base_commit and EDITABLE-INSTALLED into a conda env (`testbed`)
# with every dependency, plus build artifacts (*.egg-info, pytest _version.py,
# compiled extensions). Our host git-worktree path lacked all of that, so an
# in-episode `import <pkg>` or test run died on missing deps -> the whole N=5
# battery was deprecated. This wires each episode INSIDE the official image.
#
# PATTERN (documented design choice): WORKSPACE-MOUNT + DOCKER-EXEC hybrid.
#   * The image's /testbed is the ground-truth runtime; we do NOT rebuild it. We
#     SEED a host workspace from the image's /testbed (`docker cp`) so the
#     editable install + build artifacts are preserved verbatim, then bind-MOUNT
#     that workspace back over /testbed in a long-lived per-instance container.
#     qwen-code stays on the HOST but its CWD is an isolated directory containing
#     only AGENTS.md. It receives no native file/edit/search tools.
#   * Every inspection, edit, and test is routed INTO the container through its
#     sole declared tool, run_shell_command, via `docker exec` with the
#     official conda-activation preamble (`conda activate testbed; cd /testbed`),
#     so imports / builds / tests use the prepared per-instance environment.
#     For a live qwen-code episode a `bash` PATH-shim (see _write_shell_shim)
#     forwards run_shell_command into that same `docker exec`.
#   * Patch = tracked `git diff base_commit` over the shared /testbed tree.
# Rejected alternative: bind-mount a BARE git checkout over /testbed -> shadows
# the editable install + build artifacts -> exactly the "troubled env" the
# directive rejects. Seeding the mount from the image is what avoids that.
# ---------------------------------------------------------------------------
CONTAINER_TESTBED = "/testbed"
# Byte-for-byte the swebench eval-script preamble (conda + locale + cd).
CONTAINER_ACTIVATE = (
    "source /opt/miniconda3/bin/activate testbed && cd /testbed && "
    "export LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8"
)


def _docker_base() -> list[str]:
    """Docker CLI prefix. `docker` where the docker group is active; override via
    SWE_DOCKER_CMD (e.g. 'sudo -A docker' with SUDO_ASKPASS) where it is not."""
    return shlex.split(os.environ.get("SWE_DOCKER_CMD", "docker"))


def _docker_arch() -> str:
    import platform
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def _container_image_for(instance_id: str) -> str:
    """Official image reference, using a certified digest where one is recorded."""
    resolved = resolve_image(instance_id)
    if not resolved["pinned"] and os.environ.get("ALLOW_UNPINNED_SWE_IMAGE") != "1":
        raise RuntimeError(
            f"no certified image digest for {instance_id}; "
            "set ALLOW_UNPINNED_SWE_IMAGE=1 to opt in"
        )
    return str(resolved["reference"])


def _drun(argv: list[str], *, timeout: int = 600, check: bool = False,
          text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(_docker_base() + argv, capture_output=True, text=text,
                          timeout=timeout, check=check)


def _image_present(image: str) -> bool:
    return _drun(["image", "inspect", image], timeout=60).returncode == 0


def _seed_workspace_from_image(*, image: str, workspace: Path) -> None:
    """Materialise the image's /testbed onto the host so it can be bind-mounted
    back (preserving the editable install + build artifacts). `docker cp` from a
    created-but-never-started throwaway container; the real container mounts the
    result over /testbed."""
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    cid = _drun(["create", image, "sleep", "3600"], timeout=120, check=True).stdout.strip()
    try:
        cp = _drun(["cp", f"{cid}:{CONTAINER_TESTBED}/.", str(workspace)], timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(f"docker cp seed failed: {cp.stderr[-400:]}")
    finally:
        _drun(["rm", "-f", cid], timeout=60)


def _start_container(*, name: str, image: str, workspace: Path) -> None:
    """Long-lived per-instance container with the seeded workspace bind-mounted
    over /testbed and networking disabled. chown the mount to the host uid (via
    the container's own root) for artifact collection; needs no host sudo."""
    _drun(["rm", "-f", name], timeout=60)
    memory = os.environ.get("SWE_CONTAINER_MEMORY", "6g")
    memory_swap = os.environ.get("SWE_CONTAINER_MEMORY_SWAP", "8g")
    pids_limit = os.environ.get("SWE_CONTAINER_PIDS_LIMIT", "1024")
    run = _drun(["run", "-d", "--network", "none",
                 "--memory", memory, "--memory-swap", memory_swap,
                 "--pids-limit", pids_limit, "--name", name,
                 "-v", f"{workspace}:{CONTAINER_TESTBED}",
                 image, "sleep", "infinity"], timeout=120)
    if run.returncode != 0:
        raise RuntimeError(f"docker run failed: {run.stderr[-400:]}")
    _cexec(name, f"chown -R {os.getuid()}:{os.getgid()} {CONTAINER_TESTBED} "
                 f"&& git config --global --add safe.directory {CONTAINER_TESTBED}",
           activate=False, timeout=180)


def _stop_container(name: str) -> None:
    _drun(["rm", "-f", name], timeout=120)


def _teardown_container(name: str, workspace: Path | None = None) -> None:
    """Robust teardown: chown the bind-mounted /testbed back to the host uid FIRST
    (the container runs as root, so builds/tests leave root-owned __pycache__/
    egg-info the host user otherwise cannot delete), THEN remove the container and
    the seeded workspace. Idempotent + never raises."""
    try:
        _cexec(name, f"chown -R {os.getuid()}:{os.getgid()} {CONTAINER_TESTBED}",
               activate=False, timeout=180)
    except Exception:  # noqa: BLE001
        pass
    _stop_container(name)
    if workspace is not None:
        shutil.rmtree(workspace, ignore_errors=True)


def _cexec(name: str, command: str, *, activate: bool = True, workdir: str = CONTAINER_TESTBED,
           timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a shell command INSIDE the per-instance container. When activate=True
    the official conda `testbed` env + locale + cd /testbed preamble is prepended
    (this is what makes in-episode imports / tests use the aligned runtime)."""
    inner = f"{CONTAINER_ACTIVATE} && {command}" if activate else command
    return _drun(["exec", "-w", workdir, name, "/bin/bash", "-lc", inner], timeout=timeout)


def _container_extract_patch(name: str, base_commit: str) -> str:
    """Tracked diff vs base_commit computed INSIDE the container (shares /testbed
    with the host mount, so identical to a host diff but avoids ownership races)."""
    commit = _validated_commit(base_commit)
    cp = _cexec(name, f"git -c core.fileMode=false diff --no-color --binary {shlex.quote(commit)}",
                activate=False, timeout=300)
    return cp.stdout


def _write_shell_shim(*, shim_dir: Path, container: str) -> Path:
    """Write a `bash` shim (first on PATH) that forwards qwen-code's
    run_shell_command (`bash -c "<cmd>"`) into `docker exec` on the per-instance
    container, conda-activated. The certified wrapper's only agent tool is this
    shell path, so inspection, edits, builds, and tests all run there. Non-`-c`
    invocations fall through to the real bash."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "bash"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, os\n"
        f"DOCKER = {json.dumps(_docker_base())}\n"
        f"CONTAINER = {json.dumps(container)}\n"
        f"PRE = {json.dumps(CONTAINER_ACTIVATE)}\n"
        "args = sys.argv[1:]\n"
        "cmd = None\n"
        "for i, a in enumerate(args):\n"
        "    if a in ('-c', '-lc', '-cl') and i + 1 < len(args):\n"
        "        cmd = args[i + 1]; break\n"
        "    if a.startswith('-') and 'c' in a and i + 1 < len(args):\n"
        "        cmd = args[i + 1]; break\n"
        "if cmd is None:\n"
        "    os.execv('/bin/bash', ['/bin/bash'] + args)\n"
        "os.execvp(DOCKER[0], DOCKER + ['exec', '-w', '/testbed', CONTAINER,\n"
        "          '/bin/bash', '-lc', PRE + ' && ' + cmd])\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _run_mock_agent_container(*, container: str, instance: dict, base_commit: str,
                              workspace: Path, trace_path: Path) -> dict[str, Any]:
    """Scripted (NO model) agent for the container runtime: replay the dataset
    gold `patch` THROUGH the container (git apply via docker exec) to prove the
    hydrate -> in-container edit -> in-container patch-extract plumbing."""
    started = time.monotonic()
    gold = instance.get("patch") or ""
    applied = False
    apply_err = ""
    if gold.strip():
        (workspace / ".mock_gold.patch").write_text(gold, encoding="utf-8")
        cp = _cexec(container, "git apply --whitespace=nowarn .mock_gold.patch",
                    activate=False, timeout=180)
        applied = cp.returncode == 0
        apply_err = (cp.stderr or "").strip()[:800]
        (workspace / ".mock_gold.patch").unlink(missing_ok=True)
    trace_path.write_text(json.dumps({
        "mock_agent": True, "runtime": "container", "action": "replay_gold_patch_in_container",
        "gold_patch_bytes": len(gold), "applied": applied, "apply_stderr": apply_err,
    }, indent=2) + "\n", encoding="utf-8")
    return {
        "elapsed_s": round(time.monotonic() - started, 3), "exit_code": 0 if applied else 1,
        "timed_out": False, "cli_exit_is_verdict": False, "parsed": True,
        "subtype": "mock_gold_replay_container", "num_turns": 1, "duration_api_ms": 0,
        "usage": None, "tool_calls": 1, "tool_by_name": {"mock_apply_in_container": 1},
        "result_tail": f"gold_replay_in_container applied={applied}", "mock_apply_err": apply_err,
    }


# ---------------------------------------------------------------------------
# Qwen Code proxy lifecycle (Stage-A qwen_code_sglang_proxy.py adapter).
# ---------------------------------------------------------------------------
def _wait_proxy(url: str, token: str, proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                payload = json.load(resp)
                if resp.status == 200 and payload.get("token") == token:
                    return True
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(0.25)
    return False


def _check_upstream(
    endpoint: str, timeout: float, model: str, max_model_len: int
) -> tuple[bool, str]:
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.load(resp)
            return validate_models_payload(payload, model, max_model_len)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _start_proxy(args, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable, str(args.proxy_script),
        "--host", args.proxy_host,
        "--port", str(args.proxy_port),
        "--upstream", args.endpoint,
        "--max-tokens", str(args.proxy_max_tokens),
        "--context-limit", str(args.proxy_context_limit),
    ]
    if args.proxy_tool_choice:
        cmd += ["--tool-choice", args.proxy_tool_choice]
    if args.proxy_dump_dir:
        cmd += ["--dump-dir", str(args.proxy_dump_dir)]
    if args.proxy_required_tools is not None:
        cmd += ["--required-tools", args.proxy_required_tools]
    health_token = secrets.token_urlsafe(24)
    proxy_env = os.environ.copy()
    proxy_env["LUMO_PROXY_HEALTH_TOKEN"] = health_token
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        text=True, env=proxy_env,
    )
    if not _wait_proxy(
        f"http://{args.proxy_host}:{args.proxy_port}/health", health_token, proc, 15
    ):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        raise RuntimeError(f"proxy did not become healthy; see {log_path}")
    return proc, log_file


def _stop_proxy(proc, log_file) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_file is not None:
        log_file.close()


# ---------------------------------------------------------------------------
# Qwen Code episode runner (replaces the flywheel's `_run_codex`).
# ---------------------------------------------------------------------------
def _parse_qwen_result(stdout: str) -> dict[str, Any]:
    """Parse the qwen `--output-format json` output into a compact summary.

    Qwen Code emits either a single result object or a list of events; the final
    result object carries `subtype` (success | error_during_execution), `num_turns`,
    `duration_ms`/`duration_api_ms`, `usage` (token counts), `stats.tools`, and
    `result` (final text). Robust to either shape."""
    out: dict[str, Any] = {
        "parsed": False, "subtype": None, "num_turns": None,
        "duration_api_ms": None, "usage": None,
        "tool_calls": None, "tool_by_name": None, "result_tail": "",
    }
    stripped = (stdout or "").strip()
    if not stripped:
        return out
    obj = None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        # try last JSON line
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if obj is None:
        return out
    events = obj if isinstance(obj, list) else [obj]
    out["parsed"] = True
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("subtype"):
            out["subtype"] = ev.get("subtype")
        for k in ("num_turns",):
            if ev.get(k) is not None:
                out[k] = ev.get(k)
        for k in ("duration_api_ms", "durationApiMs", "duration_ms", "durationMs"):
            if ev.get(k) is not None:
                out["duration_api_ms"] = ev.get(k)
        if isinstance(ev.get("usage"), dict):
            out["usage"] = ev["usage"]
        stats = ev.get("stats") if isinstance(ev.get("stats"), dict) else None
        tools = ev.get("tools") if isinstance(ev.get("tools"), dict) else (
            stats.get("tools") if stats and isinstance(stats.get("tools"), dict) else None)
        if isinstance(tools, dict):
            out["tool_calls"] = tools.get("totalCalls", out["tool_calls"])
            out["tool_by_name"] = tools.get("byName", out["tool_by_name"])
        if isinstance(ev.get("result"), str) and ev["result"].strip():
            out["result_tail"] = ev["result"].strip()[-600:]
    return out


def _run_qwen_code(
    *,
    workspace: Path,
    proxy_base_url: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    args,
    trace_path: Path,
    stderr_path: Path,
    prompt: str = DEFAULT_AGENT_PROMPT,
    extra_path: Path | None = None,
) -> dict[str, Any]:
    """Run the local Qwen Code CLI headless with CWD=workspace, capturing the
    --output-format json event stream to trace_path and CLI stderr to stderr_path.

    Mirrors the Stage-A-proven invocation (runs/stage_a_smoke) + the flywheel
    QWEN_CODE_TEMPLATE budgets (--max-session-turns, MAX_OUTPUT_TOKENS).

    extra_path: dir prepended to PATH (the container runtime passes the `bash`
    shell-shim dir here so run_shell_command routes into `docker exec`)."""
    # Clean, writable, per-episode HOME (flywheel HOME=/tmp convention). REQUIRED now
    # that we drop --bare: qwen-code auto-discovers ~/.qwen config/state from HOME, so we
    # isolate it from the host user's global config AND guarantee a writable ~/.qwen for
    # the CLI session. AGENTS.md task-context is still auto-discovered from CWD (workspace).
    home_dir = trace_path.parent / "qwen_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    qwen_config = home_dir / ".qwen"
    qwen_config.mkdir(parents=True, exist_ok=True)
    (qwen_config / "settings.json").write_text(
        json.dumps({"tools": {"computerUse": {"enabled": False}}}) + "\n",
        encoding="utf-8",
    )
    # Do not give an unattended, prompt-driven agent the operator's credentials.
    # The pinned task needs only local process basics plus the explicit model env.
    safe_keys = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ")
    env = {key: os.environ[key] for key in safe_keys if key in os.environ}
    # Flywheel qwen-code env profile (QWEN_CODE_TEMPLATE / instance_image path): auth +
    # endpoint + model are supplied via ENV (NOT CLI flags), plus the context-budget and
    # stream-idle timeouts. OPENAI_API_KEY is the flywheel's non-empty "EMPTY" placeholder
    # (the proxy ignores auth). An existing QWEN_STREAM_IDLE_TIMEOUT_MS env wins so gate
    # scripts can bump it (e.g. 600000 for batched B>=4) without editing the driver.
    env.update({
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": proxy_base_url,
        "OPENAI_MODEL": model,
        "QWEN_MODEL": model,
        "QWEN_CODE_MAX_OUTPUT_TOKENS": str(args.qwen_max_output_tokens),
        "QWEN_STREAM_IDLE_TIMEOUT_MS": str(args.qwen_stream_idle_timeout_ms),
        "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
        # Qwen 0.19.4 treats this as an unconditional computer-use disable,
        # even if a project-level settings file attempts to enable it.
        "QWEN_SERVE_CDP_TUNNEL_OVER_WS": "1",
        "NO_BROWSER": "1",
        "NO_COLOR": "1",
        "HOME": str(home_dir),
    })
    if extra_path is not None:
        env["PATH"] = str(extra_path) + os.pathsep + env.get("PATH", "")
    # Qwen Code 0.19.4 invocation. Auth/endpoint/model flow via ENV (above), not
    # CLI flags. The caller supplies the one-item core-tool allowlist, disables
    # extension discovery, and uses an isolated CWD so AGENTS.md remains visible
    # without exposing the mounted repository to native host-side tools.
    cmd = [
        str(args.qwen_bin),
        "--yolo",
        "--output-format", "json",
        "--max-session-turns", str(args.max_session_turns),
        "--extensions", "__lumo_disabled__",
    ]
    # Optional hard qwen run-level wall (flywheel default: none -> rely on the stream-idle
    # timeout). Opt-in via --qwen-max-wall "1800s" (exit 55 on overrun).
    if args.qwen_max_wall:
        cmd += ["--max-wall-time", args.qwen_max_wall]
    # Core-tool allowlist and defense-in-depth exclusions come from the certified
    # run configuration.
    for tool in (getattr(args, "core_tools", None) or []):
        cmd += ["--core-tools", tool]
    for tool in (getattr(args, "exclude_tools", None) or []):
        cmd += ["--exclude-tools", tool]
    if args.system_prompt:
        cmd += ["--system-prompt", args.system_prompt]
    cmd += ["-p", prompt]

    started = time.monotonic()
    rc: int | None = None
    timed_out = False
    with trace_path.open("w", encoding="utf-8") as tf, stderr_path.open("w", encoding="utf-8") as ef:
        try:
            completed = subprocess.run(
                cmd, cwd=str(workspace), stdout=tf, stderr=ef, env=env,
                timeout=(None if timeout_s <= 0 else max(timeout_s, 30)),
                check=False,
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -1
    elapsed = time.monotonic() - started
    parsed = _parse_qwen_result(trace_path.read_text(errors="replace") if trace_path.is_file() else "")
    return {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        # R4: exit_code is DIAGNOSTIC ONLY — 1 (loop-detector halt) / 55 (budget)
        # are NOT failures if the patch/eval says otherwise.
        "cli_exit_is_verdict": False,
        **{k: parsed[k] for k in ("parsed", "subtype", "num_turns", "duration_api_ms",
                                  "usage", "tool_calls", "tool_by_name", "result_tail")},
    }


def _run_mock_agent(*, workspace: Path, instance: dict, trace_path: Path) -> dict[str, Any]:
    """Dry-run agent (NO model server): replay the dataset gold `patch` into the
    worktree via `git apply`. Proves the hydrate -> edit -> patch-extract plumbing
    end-to-end deterministically. Records what it did to trace_path."""
    started = time.monotonic()
    gold = instance.get("patch") or ""
    applied = False
    apply_err = ""
    if gold.strip():
        pf = workspace / ".mock_gold.patch"
        pf.write_text(gold, encoding="utf-8")
        proc = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", str(pf)],
            capture_output=True, text=True, check=False,
        )
        applied = proc.returncode == 0
        apply_err = proc.stderr.strip()[:800]
        pf.unlink(missing_ok=True)
    trace_path.write_text(json.dumps({
        "mock_agent": True, "action": "replay_gold_patch",
        "gold_patch_bytes": len(gold), "applied": applied, "apply_stderr": apply_err,
    }, indent=2) + "\n", encoding="utf-8")
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "exit_code": 0 if applied else 1,
        "timed_out": False, "cli_exit_is_verdict": False,
        "parsed": True, "subtype": "mock_gold_replay",
        "num_turns": 1, "duration_api_ms": 0, "usage": None,
        "tool_calls": 1, "tool_by_name": {"mock_apply": 1},
        "result_tail": f"gold_replay applied={applied}",
        "mock_apply_err": apply_err,
    }


# ---------------------------------------------------------------------------
# Empty-patch classification + state-conditional retry (ported from flywheel;
# default retries 0 = nudge-only, dormant unless SWE_EMPTY_PATCH_RETRIES>=1).
# ---------------------------------------------------------------------------
def _classify_empty_patch_cause(trace_path: Path) -> str:
    """>=3 identical failing shell commands => setup_loop, else agent_gave_up.
    Reads the qwen json event stream (tool call args)."""
    try:
        text = trace_path.read_text(errors="replace") if trace_path.is_file() else ""
        cmds: Counter = Counter()
        # qwen json: shell tool calls appear as run_shell_command with a `command`.
        for tok in text.split('"command"'):
            frag = tok[:400]
            start = frag.find(":")
            if start >= 0:
                cmds[frag[start:start + 200]] += 1
        if cmds and max(cmds.values()) >= 3:
            return "setup_loop"
    except Exception:  # noqa: BLE001
        pass
    return "agent_gave_up"


# ---------------------------------------------------------------------------
# Terminal-cause classification (HARNESS TRUTH-TELLING fix, 2026-07-12).
#
# When the proxy's context-overflow retry ladder EXHAUSTS (halves max_tokens
# 8127→…→253 while the prompt stays pinned at the 32516 cap and every rung lands
# one token over — the 32516/253/32769 cap+1 signature documented in
# runs/k_gate_c46/K1_COMMITTAL_ANALYSIS.md "Terminal trigger"), the proxy forwards
# vLLM's final HTTP 400 to qwen-code, which surfaces it as the episode's TERMINAL
# `result` payload ("[API Error: 400 This model's maximum context length is N
# tokens. …]"). Such an episode is NOT a voluntary exit-0 quit and NOT an honest
# empty-patch miss — it is ENV-LIMITED (the served context window is exhausted).
# We tag it here so the ledger + gate report builders can map it to env-limited
# instead of empty_patch-as-honest-miss / exit-0-clean (the mislabel the AR_PAIRED
# read called out). This is a labeling fix ONLY — it does NOT change the retry
# ladder, the clamp shim, or any decode piece.
_CTX_OVERFLOW_RE = re.compile(r"maximum context length is\s+\d+\s+tokens", re.IGNORECASE)
TERMINAL_CTX_OVERFLOW = "ctx_overflow"


def _result_is_ctx_overflow(result_tail: str | None) -> bool:
    """True iff the terminal `result` payload is vLLM's context-length 400.

    Requires BOTH the "maximum context length is N tokens" phrase AND the upstream
    API-error framing (qwen-code emits "[API Error: 400 …]"), so a model that merely
    QUOTES the phrase in a normal summary is not misclassified as an env-limited death.
    """
    if not result_tail:
        return False
    txt = result_tail
    return bool(_CTX_OVERFLOW_RE.search(txt)) and ("API Error" in txt or "400" in txt)


def _agent_metas(summary: dict) -> list[dict]:
    """The ordered agent attempts on this episode: the main drive followed by any
    empty-patch re-drives (`qwen_retry1`, `qwen_retry2`, …). The LAST attempt that
    produced a terminal result is the episode's true terminal event."""
    metas: list[dict] = []
    q = summary.get("qwen")
    if isinstance(q, dict):
        metas.append(q)
    for k in sorted(k for k in summary if k.startswith("qwen_retry")):
        rv = summary.get(k)
        if isinstance(rv, dict):
            metas.append(rv)
    return metas


def _classify_terminal_cause(agent_metas: list[dict]) -> str | None:
    """Return a DISTINGUISHABLE env-limited terminal cause, else None (a normal
    terminal — the exit-mode / patch covariates already describe it).

    Walks the attempts newest-first: the latest attempt WITH a terminal result is
    the episode's terminal event (a re-drive that produced nothing — empty
    result_tail — is skipped so we fall back to the drive that actually terminated).
    Currently only `ctx_overflow` is distinguished (the 32768-window exhaustion the
    retry ladder surfaces)."""
    for meta in reversed(agent_metas):
        if not isinstance(meta, dict):
            continue
        rt = meta.get("result_tail")
        if not rt:
            continue
        return TERMINAL_CTX_OVERFLOW if _result_is_ctx_overflow(rt) else None
    return None


# ---------------------------------------------------------------------------
# Eval dispatch. Normal runs use skip, then call score_verified.sh separately.
# ---------------------------------------------------------------------------
def _emit_report(output_dir: Path, *, instance_id: str, dataset_name: str, model_name: str,
                 patch_path: Path, predictions_path: Path, verdict: str, passed: bool,
                 failure_mode: str | None, harness_exit_code: int, elapsed_s: float,
                 error: str | None, extra: dict[str, Any] | None = None) -> None:
    import platform
    report = {
        "track": "swe_bench", "instance_id": instance_id, "model_id": model_name,
        "dataset_name": dataset_name, "patch_path": str(patch_path),
        "prediction_path": str(predictions_path), "verdict": verdict, "passed": passed,
        "failure_mode": failure_mode, "harness_exit_code": harness_exit_code,
        "eval_wall_clock_seconds": round(elapsed_s, 3), "error": error,
    }
    if extra:
        report.update(extra)
    (output_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "normalized_eval.json").write_text(json.dumps({
        "track": "swe_bench", "instance_id": instance_id, "outcome": verdict,
        "failure_mode": failure_mode, "dataset_name": dataset_name, "model_id": model_name,
        "eval_wall_clock_seconds": round(elapsed_s, 3), "arch": platform.machine().lower(),
    }, indent=2))


def _write_predictions(predictions_path: Path, *, instance_id: str, patch_text: str,
                       model_name: str) -> None:
    predictions_path.write_text(json.dumps({
        "instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch_text,
    }) + "\n")


def _changed_lines(p: str) -> set[str]:
    """The multiset (as a set) of added/removed *content* lines of a diff,
    excluding file headers (+++/---) and hunk headers. Robust to hunk-offset /
    context-line drift between the stored gold and a re-extracted git diff, while
    still capturing whether the same code change was made."""
    out: set[str] = set()
    for ln in p.splitlines():
        if ln.startswith(("+++", "---")):
            continue
        if ln.startswith(("+", "-")) and ln[1:].strip():
            out.add(ln.rstrip())
    return out


def _run_eval_mock(*, instance_id: str, instance: dict, patch_text: str, patch_path: Path,
                   output_dir: Path, dataset_name: str, model_name: str) -> dict[str, Any]:
    """No-docker stand-in for the swebench harness (dry-run plumbing only).

    NOT a real score: resolved iff the extracted patch equals the dataset gold
    code patch (normalized); empty -> patch_apply_failed; else -> tests_failed.
    Clearly labelled `mock=True` so it can never be confused with a docker run."""
    started = time.monotonic()
    predictions_path = output_dir / "predictions.jsonl"
    _write_predictions(predictions_path, instance_id=instance_id, patch_text=patch_text,
                       model_name=model_name)
    if not patch_text.strip():
        verdict, passed, fmode, rc = "failed", False, "patch_apply_failed", 1
    else:
        gold = instance.get("patch") or ""
        gold_lines = _changed_lines(gold)
        got_lines = _changed_lines(patch_text)
        if gold_lines and gold_lines.issubset(got_lines):
            verdict, passed, fmode, rc = "resolved", True, "tests_passed", 0
        else:
            verdict, passed, fmode, rc = "failed", False, "tests_failed", 1
    _emit_report(output_dir, instance_id=instance_id, dataset_name=dataset_name,
                 model_name=model_name, patch_path=patch_path, predictions_path=predictions_path,
                 verdict=verdict, passed=passed, failure_mode=fmode, harness_exit_code=rc,
                 elapsed_s=time.monotonic() - started, error=None,
                 extra={"mock": True, "eval_backend": "mock_gold_compare"})
    (output_dir / "eval.log").write_text(
        f"[MOCK EVAL — NOT a docker harness run] verdict={verdict} "
        f"patch_bytes={len(patch_text)} gold_bytes={len(instance.get('patch') or '')}\n",
        encoding="utf-8")
    return {"exit_code": rc, "elapsed_s": round(time.monotonic() - started, 3), "backend": "mock"}


_EVAL_SSH_OPTS = [
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
    "-o", "StrictHostKeyChecking=accept-new",
]
_REMOTE_BASE = "~/swe_eval_offload"
_REMOTE_WORKER = "~/swe_eval_offload/swe_eval_x86_worker.py"
_REMOTE_VENV_PY = "~/swe_eval_offload/venv/bin/python"
_REMOTE_HF_HOME = "~/.cache/huggingface"


def _run_eval_offload(*, host: str, instance_id: str, patch_path: Path, output_dir: Path,
                      dataset_name: str, model_name: str, timeout_s: int,
                      eval_log_path: Path) -> dict[str, Any]:
    """Offload eval to a native x86_64 box over SSH (plan §C3; ported from the
    flywheel `_run_eval_remote`). The remote runs swe_eval_x86_worker.py and we
    fetch the flywheel artifact set back."""
    started = time.monotonic()
    remote_dir = f"{_REMOTE_BASE}/work/{instance_id}"

    def _ssh(argv, timeout):
        return subprocess.run(["ssh", *_EVAL_SSH_OPTS, host, argv], capture_output=True,
                              text=True, timeout=timeout)

    mk = _ssh(f"mkdir -p {remote_dir} && echo ok", 30)
    if mk.returncode != 0:
        eval_log_path.write_text(f"remote mkdir failed rc={mk.returncode}\n{mk.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "backend": "offload"}
    up = subprocess.run(["scp", *_EVAL_SSH_OPTS, str(patch_path), f"{host}:{remote_dir}/patch.diff"],
                        capture_output=True, text=True, timeout=120)
    if up.returncode != 0:
        eval_log_path.write_text(f"scp up failed rc={up.returncode}\n{up.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "backend": "offload"}
    remote_cmd = (
        f"cd {_REMOTE_BASE} && HF_HOME={_REMOTE_HF_HOME} {_REMOTE_VENV_PY} {_REMOTE_WORKER} "
        f"--instance-id {instance_id} --patch-path {remote_dir}/patch.diff "
        f"--output-dir {remote_dir}/out --dataset-name '{dataset_name}' "
        f"--model-name '{model_name}' --timeout-s {timeout_s} --cache-level env"
    )
    ev = _ssh(remote_cmd, timeout_s + 900)
    output_dir.mkdir(parents=True, exist_ok=True)
    with eval_log_path.open("w", encoding="utf-8") as f:
        f.write(f"[offload host={host} worker_rc={ev.returncode}]\n{ev.stdout}\n-- stderr --\n{ev.stderr}")
    for fname in ("eval_report.json", "normalized_eval.json", "eval.log", "predictions.jsonl"):
        subprocess.run(["scp", *_EVAL_SSH_OPTS, f"{host}:{remote_dir}/out/{fname}",
                        str(output_dir / fname)], capture_output=True, text=True, timeout=120)
    _ssh(f"rm -rf {remote_dir}", 30)
    return {"exit_code": ev.returncode, "elapsed_s": round(time.monotonic() - started, 3),
            "backend": "offload", "eval_host": host}


def _run_eval_local(*, instance_id: str, patch_path: Path, output_dir: Path, dataset_name: str,
                    model_name: str, timeout_s: int, eval_log_path: Path) -> dict[str, Any]:
    raise RuntimeError("use scripts/score_verified.sh for official local scoring")


def _run_eval(*, mode: str, eval_host: str | None, instance_id: str, instance: dict,
              patch_text: str, patch_path: Path, output_dir: Path, dataset_name: str,
              model_name: str, timeout_s: int, eval_log_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "skip":
        _write_predictions(output_dir / "predictions.jsonl", instance_id=instance_id,
                           patch_text=patch_text, model_name=model_name)
        _emit_report(output_dir, instance_id=instance_id, dataset_name=dataset_name,
                     model_name=model_name, patch_path=patch_path,
                     predictions_path=output_dir / "predictions.jsonl",
                     verdict="skipped", passed=False, failure_mode=None, harness_exit_code=0,
                     elapsed_s=0.0, error=None, extra={"eval_backend": "skip"})
        return {"exit_code": 0, "elapsed_s": 0.0, "backend": "skip"}
    if mode == "mock":
        return _run_eval_mock(instance_id=instance_id, instance=instance, patch_text=patch_text,
                              patch_path=patch_path, output_dir=output_dir,
                              dataset_name=dataset_name, model_name=model_name)
    if mode == "offload":
        if not eval_host:
            raise SystemExit("--eval-mode offload requires --eval-host")
        return _run_eval_offload(host=eval_host, instance_id=instance_id, patch_path=patch_path,
                                 output_dir=output_dir, dataset_name=dataset_name,
                                 model_name=model_name, timeout_s=timeout_s,
                                 eval_log_path=eval_log_path)
    if mode == "local":
        return _run_eval_local(instance_id=instance_id, patch_path=patch_path,
                               output_dir=output_dir, dataset_name=dataset_name,
                               model_name=model_name, timeout_s=timeout_s,
                               eval_log_path=eval_log_path)
    raise SystemExit(f"unknown eval mode: {mode}")


# ---------------------------------------------------------------------------
# Per-instance orchestration.
# ---------------------------------------------------------------------------
def _process_one(*, instance_id: str, instance: dict, dataset_name: str, per_task_root: Path,
                 repo_cache_root: Path, proxy_base_url: str | None, model: str, model_name: str,
                 agent: str, agent_wall_s: int, eval_mode: str, eval_host: str | None,
                 eval_timeout_s: int, skip_existing: bool, args) -> dict[str, Any]:
    task_dir = _safe_task_dir(per_task_root, instance_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    qwen_trace = task_dir / "qwen_trace.json"
    qwen_stderr = task_dir / "qwen_stderr.log"
    prompt_md = task_dir / "prompt.md"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)

    base_commit = _validated_commit(instance["base_commit"])
    summary: dict[str, Any] = {
        "instance_id": instance_id, "dataset_name": dataset_name, "started_at": _iso_now(),
        "repo": instance.get("repo"), "base_commit": base_commit,
        "agent": agent, "eval_mode": eval_mode,
    }

    cache_path = None
    try:
        cache_path = _ensure_repo_cache(instance["repo"], repo_cache_root)
        _fetch_commit(cache_path, base_commit)
        _hydrate_workspace(cache_path=cache_path, base_commit=base_commit,
                           workspace_path=workspace_path)
        _write_agents_md(workspace_path, instance)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        if cache_path is not None:
            _remove_workspace(cache_path, workspace_path)
        return summary

    prompt_md.write_text(
        "## Qwen Code invocation prompt\n\n" + DEFAULT_AGENT_PROMPT + "\n\n"
        f"## AGENTS.md (workspace/{instance_id})\n\n"
        + (workspace_path / "AGENTS.md").read_text(encoding="utf-8"), encoding="utf-8")

    # --- agent episode -----------------------------------------------------
    if agent == "mock":
        agent_meta = _run_mock_agent(workspace=workspace_path, instance=instance,
                                     trace_path=qwen_trace)
    else:
        agent_meta = _run_qwen_code(workspace=workspace_path, proxy_base_url=proxy_base_url,
                                    model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                    args=args, trace_path=qwen_trace, stderr_path=qwen_stderr)
    summary["qwen"] = agent_meta

    # --- patch extraction --------------------------------------------------
    patch_text = ""
    try:
        patch_text = _extract_patch(workspace_path, instance["base_commit"])
    except Exception as exc:  # noqa: BLE001
        summary["patch_extract_error"] = f"{type(exc).__name__}: {exc}"

    # --- optional state-conditional empty-patch retry (default 0) ----------
    if not patch_text.strip() and agent != "mock":
        cause = _classify_empty_patch_cause(qwen_trace)
        max_retries = max(0, int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")))
        summary["empty_patch_retry"] = {"cause": cause, "max_retries": max_retries,
                                        "recovered_patch_bytes": 0}
        for ridx in range(1, max_retries + 1):
            retry_prompt = RETRY_PROMPT_SETUP_LOOP if cause == "setup_loop" else RETRY_PROMPT_EMPTY
            rtrace = task_dir / f"qwen_trace_retry{ridx}.json"
            rstderr = task_dir / f"qwen_stderr_retry{ridx}.log"
            rmeta = _run_qwen_code(workspace=workspace_path, proxy_base_url=proxy_base_url,
                                   model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                   args=args, trace_path=rtrace, stderr_path=rstderr,
                                   prompt=retry_prompt)
            summary[f"qwen_retry{ridx}"] = rmeta
            try:
                rp = _extract_patch(workspace_path, instance["base_commit"])
            except Exception:  # noqa: BLE001
                rp = ""
            if rp.strip():
                patch_text = rp
                summary["empty_patch_retry"]["recovered_patch_bytes"] = len(rp)
                break

    patch_path.write_text(patch_text, encoding="utf-8")
    summary["patch_bytes"] = len(patch_text)

    # --- eval (verdict source of truth; NOT the qwen CLI exit — R4) --------
    eval_meta = _run_eval(mode=eval_mode, eval_host=eval_host, instance_id=instance_id,
                          instance=instance, patch_text=patch_text, patch_path=patch_path,
                          output_dir=eval_output, dataset_name=dataset_name, model_name=model_name,
                          timeout_s=eval_timeout_s, eval_log_path=eval_log)
    summary["eval"] = eval_meta
    report_path = eval_output / "eval_report.json"
    if report_path.is_file():
        try:
            summary["eval_report"] = json.loads(report_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    _remove_workspace(cache_path, workspace_path)
    # Truth-telling terminal-cause tag (None = normal terminal; "ctx_overflow" =
    # the context-window-exhaustion death the retry ladder surfaces). Consumed by
    # ledger.record + the gate report builders so a cap-death is NOT recorded as an
    # honest empty-patch / exit-0-clean.
    summary["terminal_cause"] = _classify_terminal_cause(_agent_metas(summary))
    summary["ended_at"] = _iso_now()
    runner_meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def _process_one_container(*, instance_id: str, instance: dict, dataset_name: str,
                           per_task_root: Path, proxy_base_url: str | None, model: str,
                           model_name: str, agent: str, agent_wall_s: int, eval_mode: str,
                           eval_host: str | None, eval_timeout_s: int, skip_existing: bool,
                           args) -> dict[str, Any]:
    """CONTAINER runtime (RUNTIME_ALIGNMENT_DIRECTIVE): episode runs inside the
    official per-instance swebench image. Same artifact set as `_process_one`;
    the workspace is seeded from the image /testbed and bind-mounted into a
    long-lived container, the agent's shell routes through `docker exec`, and the
    patch is the in-container tracked diff vs base_commit."""
    task_dir = _safe_task_dir(per_task_root, instance_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    qwen_trace = task_dir / "qwen_trace.json"
    qwen_stderr = task_dir / "qwen_stderr.log"
    prompt_md = task_dir / "prompt.md"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)
    shim_dir = task_dir / "shim"
    agent_cwd = task_dir / "agent_cwd"

    base_commit = _validated_commit(instance["base_commit"])
    image = _container_image_for(instance_id)
    container = f"{args.container_name_prefix}_{instance_id.replace('__', '_').replace('/', '_')}"

    summary: dict[str, Any] = {
        "instance_id": instance_id, "dataset_name": dataset_name, "started_at": _iso_now(),
        "repo": instance.get("repo"), "base_commit": base_commit, "agent": agent,
        "eval_mode": eval_mode, "runtime": "container", "image": image, "container": container,
    }

    try:
        if not _image_present(image):
            raise RuntimeError(f"official image not present locally: {image} "
                               f"(docker pull {image})")
        _seed_workspace_from_image(image=image, workspace=workspace_path)
        _start_container(name=container, image=image, workspace=workspace_path)
        _write_agents_md(workspace_path, instance)
        if agent_cwd.exists():
            shutil.rmtree(agent_cwd)
        agent_cwd.mkdir()
        shutil.copy2(workspace_path / "AGENTS.md", agent_cwd / "AGENTS.md")
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        _teardown_container(container, workspace_path if not args.container_keep else None)
        return summary

    prompt_md.write_text(
        "## Qwen Code invocation prompt (container runtime)\n\n" + DEFAULT_AGENT_PROMPT + "\n\n"
        f"## AGENTS.md (workspace/{instance_id})\n\n"
        + (workspace_path / "AGENTS.md").read_text(encoding="utf-8"), encoding="utf-8")

    def _extract() -> str:
        try:
            return _container_extract_patch(container, base_commit)
        except Exception:  # noqa: BLE001
            return ""

    try:
        # --- agent episode (edits land in the container via mount + docker exec)
        if agent == "mock":
            agent_meta = _run_mock_agent_container(container=container, instance=instance,
                                                   base_commit=base_commit,
                                                   workspace=workspace_path, trace_path=qwen_trace)
        else:
            _write_shell_shim(shim_dir=shim_dir, container=container)
            agent_meta = _run_qwen_code(workspace=agent_cwd, proxy_base_url=proxy_base_url,
                                        model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                        args=args, trace_path=qwen_trace, stderr_path=qwen_stderr,
                                        extra_path=shim_dir)
        summary["qwen"] = agent_meta

        patch_text = _extract()

        # --- re-drive mitigation: state-conditional empty-patch retry ----------
        # The historical source envelope (temperature 0.6) had a documented tool-call-free-terminal
        # flake: the agent EXPLORES then emits a text-only terminal reply with no
        # edit (flywheel run_swe_bench_q36_a.py:976 area). When the episode leaves
        # NO patch, classify why (setup_loop vs agent_gave_up) and RE-DRIVE the
        # agent up to N times with a hard must-act directive. N comes from
        # --empty-patch-retries (env SWE_EMPTY_PATCH_RETRIES fallback); default 0 =
        # OFF (byte-identical to a plain single attempt), the envelope run sets >=1.
        if not patch_text.strip() and agent != "mock":
            cause = _classify_empty_patch_cause(qwen_trace)
            max_retries = _empty_patch_retries(args)
            summary["empty_patch_retry"] = {"cause": cause, "max_retries": max_retries,
                                            "recovered_patch_bytes": 0}
            for ridx in range(1, max_retries + 1):
                retry_prompt = RETRY_PROMPT_SETUP_LOOP if cause == "setup_loop" else RETRY_PROMPT_EMPTY
                rmeta = _run_qwen_code(workspace=agent_cwd, proxy_base_url=proxy_base_url,
                                       model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                       args=args, trace_path=task_dir / f"qwen_trace_retry{ridx}.json",
                                       stderr_path=task_dir / f"qwen_stderr_retry{ridx}.log",
                                       prompt=retry_prompt, extra_path=shim_dir)
                summary[f"qwen_retry{ridx}"] = rmeta
                rp = _extract()
                if rp.strip():
                    patch_text = rp
                    summary["empty_patch_retry"]["recovered_patch_bytes"] = len(rp)
                    break

        patch_path.write_text(patch_text, encoding="utf-8")
        summary["patch_bytes"] = len(patch_text)

        eval_meta = _run_eval(mode=eval_mode, eval_host=eval_host, instance_id=instance_id,
                              instance=instance, patch_text=patch_text, patch_path=patch_path,
                              output_dir=eval_output, dataset_name=dataset_name,
                              model_name=model_name, timeout_s=eval_timeout_s, eval_log_path=eval_log)
        summary["eval"] = eval_meta
        report_path = eval_output / "eval_report.json"
        if report_path.is_file():
            try:
                summary["eval_report"] = json.loads(report_path.read_text())
            except Exception:  # noqa: BLE001
                pass
    finally:
        if not args.container_keep:
            _teardown_container(container, workspace_path)

    # Truth-telling terminal-cause tag (see _process_one): distinguishes a
    # context-window-exhaustion cap-death from an honest empty-patch / clean exit.
    summary["terminal_cause"] = _classify_terminal_cause(_agent_metas(summary))
    summary["ended_at"] = _iso_now()
    runner_meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate(per_task_root: Path, summary_path: Path, predictions_path: Path,
               started_at: str, ended_at: str, model_name: str) -> dict[str, Any]:
    """Ported from the flywheel: verdict_counts / resolved_rate / per-repo /
    wall percentiles, + predictions.jsonl concatenation."""
    instance_summaries: list[dict] = []
    verdict_counter: Counter = Counter()
    failure_counter: Counter = Counter()
    repo_counter: Counter = Counter()
    repo_pass_counter: Counter = Counter()
    agent_wall: list[float] = []
    predictions_lines: list[str] = []
    for task_dir in sorted(p for p in per_task_root.iterdir() if p.is_dir()):
        meta_path = task_dir / "runner_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        instance_summaries.append(meta)
        verdict = (meta.get("eval_report") or {}).get("verdict", "missing")
        failure = (meta.get("eval_report") or {}).get("failure_mode", "missing")
        verdict_counter[verdict] += 1
        failure_counter[failure] += 1
        repo = meta.get("repo") or "unknown"
        repo_counter[repo] += 1
        if verdict == "resolved":
            repo_pass_counter[repo] += 1
        if (meta.get("qwen") or {}).get("elapsed_s") is not None:
            agent_wall.append(float(meta["qwen"]["elapsed_s"]))
        pred_file = task_dir / "eval" / "predictions.jsonl"
        if pred_file.is_file():
            predictions_lines.extend(l for l in pred_file.read_text().splitlines() if l.strip())

    def _pcts(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        def _p(p):
            i = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
            return round(xs[i], 3)
        return {"p50": _p(0.5), "p90": _p(0.9), "min": round(min(xs), 3), "max": round(max(xs), 3)}

    summary = {
        "model_name_or_path": model_name, "started_at": started_at, "ended_at": ended_at,
        "instances_total": len(instance_summaries), "verdict_counts": dict(verdict_counter),
        "failure_mode_counts": dict(failure_counter), "per_repo_total": dict(repo_counter),
        "per_repo_resolved": dict(repo_pass_counter), "agent_wall_seconds": _pcts(agent_wall),
        "resolved_rate": (round(verdict_counter["resolved"] / len(instance_summaries), 4)
                          if instance_summaries else None),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    predictions_path.write_text("\n".join(predictions_lines) + ("\n" if predictions_lines else ""))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=Path, required=True,
                   help="JSON subset from build_swe_bench_subset.py (dataset_name + instance_ids)")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--dataset-tag", default=None)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="vLLM /v1 endpoint (proxy upstream)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served-model-name at --endpoint")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME_TAG,
                   help="tag recorded in predictions.jsonl.model_name_or_path")
    p.add_argument("--agent", choices=("qwen_code", "mock"), default="qwen_code")
    p.add_argument("--agent-wall-s", type=int,
                   default=int(os.environ.get("SWE_AGENT_WALL_S", str(DEFAULT_AGENT_WALL_S))),
                   help="hard harness wall per attempt (0 = rely on qwen --max-wall-time)")
    p.add_argument("--max-session-turns", type=int, default=DEFAULT_MAX_SESSION_TURNS,
                   help="flywheel default 100000 = no turn cap (natural submit/give-up)")
    p.add_argument("--qwen-max-wall", default=DEFAULT_QWEN_MAX_WALL,
                   help="hard qwen run-level wall (e.g. '1800s'); '' (flywheel default) "
                        "omits --max-wall-time and relies on the stream-idle timeout")
    p.add_argument("--qwen-max-output-tokens", type=int, default=DEFAULT_QWEN_MAX_OUTPUT_TOKENS)
    p.add_argument("--qwen-stream-idle-timeout-ms", type=int,
                   default=int(os.environ.get("QWEN_STREAM_IDLE_TIMEOUT_MS",
                                              str(DEFAULT_QWEN_STREAM_IDLE_TIMEOUT_MS))),
                   help="QWEN_STREAM_IDLE_TIMEOUT_MS for the qwen episode (flywheel FR13 "
                        "belt; default 240000, in-image batched path uses 600000)")
    p.add_argument("--exclude-tools", action="append", default=None,
                   help="tools to exclude from qwen-code (flywheel default: none). "
                        "Repeat for multiple, e.g. --exclude-tools web_fetch")
    p.add_argument("--core-tool", dest="core_tools", action="append", default=None,
                   help="core-tool allowlist; repeat once per permitted tool")
    p.add_argument("--system-prompt", default="")
    p.add_argument("--eval-mode", choices=("mock", "skip"), default="skip")
    p.add_argument("--eval-host", default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval-timeout-s", type=int, default=DEFAULT_EVAL_TIMEOUT_S)
    p.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    p.add_argument("--limit", type=int, default=None, help="process only first N instances")
    p.add_argument("--only", default=None, help="comma-sep instance_ids to run (subset filter)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--qwen-bin", type=Path, default=DEFAULT_QWEN_BIN)
    p.add_argument("--proxy-script", type=Path, default=DEFAULT_PROXY_SCRIPT)
    p.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    p.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    p.add_argument("--proxy-max-tokens", type=int, default=DEFAULT_PROXY_MAX_TOKENS)
    p.add_argument("--proxy-context-limit", type=int, default=32768)
    p.add_argument("--proxy-tool-choice", default="", help="'' = natural (default); 'required' forces")
    p.add_argument("--proxy-required-tools", default=None,
                   help="comma-separated exact tool-schema boundary enforced by the proxy")
    p.add_argument("--proxy-dump-dir", type=Path, default=None)
    p.add_argument("--runtime", choices=("host", "container"), default="container",
                   help="host = git-worktree bare checkout (legacy, deps absent); "
                        "container = official per-instance swebench image with the "
                        "prepared conda env (RUNTIME_ALIGNMENT_DIRECTIVE)")
    p.add_argument("--container-name-prefix", default="swe_ep",
                   help="prefix for the per-instance docker container name")
    p.add_argument("--container-keep", action="store_true",
                   help="keep the container + seeded workspace after each instance (debug)")
    p.add_argument("--empty-patch-retries", type=int, default=None,
                   help="re-drive mitigation: N re-launches with a must-act directive "
                        "when an episode leaves NO patch (the historical tool-call-free-"
                        "terminal flake). Default None -> env SWE_EMPTY_PATCH_RETRIES "
                        "-> 0 (off). The reference-envelope run sets >=1.")
    args = p.parse_args(argv)

    dataset_name, instance_ids = _load_subset(args.subset)
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        instance_ids = [i for i in instance_ids if i in want]
    if args.limit is not None:
        instance_ids = instance_ids[: args.limit]
    dataset_tag = args.dataset_tag or ("pro" if "Pro" in dataset_name else "verified")
    dataset_out = args.out_root / dataset_tag
    per_task_root = dataset_out / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)
    args.repo_cache.mkdir(parents=True, exist_ok=True)
    for iid in instance_ids:
        _safe_task_dir(per_task_root, iid)

    print(f"=== [{_iso_now()}] dataset={dataset_name} tag={dataset_tag} n={len(instance_ids)} "
          f"agent={args.agent} eval={args.eval_mode} endpoint={args.endpoint} ===", flush=True)

    # Load instance metadata (repo/base_commit/problem_statement/patch) from HF.
    dataset_records = _load_dataset(dataset_name, revision=DEFAULT_DATASET_REVISION)
    missing = [i for i in instance_ids if i not in dataset_records]
    if missing:
        raise SystemExit(f"subset instances missing from dataset: {missing[:5]}")

    # Start the qwen-code proxy once for the campaign (skip for the mock agent).
    proxy_proc = proxy_log = None
    proxy_base_url = None
    if args.agent == "qwen_code":
        if not args.qwen_bin.exists():
            raise SystemExit(f"missing qwen binary: {args.qwen_bin}")
        ok, detail = _check_upstream(
            args.endpoint, 5, args.model, args.proxy_context_limit
        )
        if not ok:
            raise SystemExit(f"upstream not ready at {args.endpoint}: {detail}")
        proxy_base_url = f"http://{args.proxy_host}:{args.proxy_port}/v1"
        proxy_proc, proxy_log = _start_proxy(args, dataset_out / "proxy.log")
        print(f"[proxy] {proxy_base_url} -> {args.endpoint}", flush=True)

    started_at = _iso_now()
    try:
        for iid in instance_ids:
            t0 = time.time()
            print(f"[{_iso_now()}] -> {iid}", flush=True)
            try:
                if args.runtime == "container":
                    res = _process_one_container(
                        instance_id=iid, instance=dataset_records[iid],
                        dataset_name=dataset_name, per_task_root=per_task_root,
                        proxy_base_url=proxy_base_url, model=args.model,
                        model_name=args.model_name, agent=args.agent,
                        agent_wall_s=args.agent_wall_s, eval_mode=args.eval_mode,
                        eval_host=args.eval_host, eval_timeout_s=args.eval_timeout_s,
                        skip_existing=args.skip_existing, args=args)
                else:
                    res = _process_one(instance_id=iid, instance=dataset_records[iid],
                                       dataset_name=dataset_name, per_task_root=per_task_root,
                                       repo_cache_root=args.repo_cache, proxy_base_url=proxy_base_url,
                                       model=args.model, model_name=args.model_name, agent=args.agent,
                                       agent_wall_s=args.agent_wall_s, eval_mode=args.eval_mode,
                                       eval_host=args.eval_host, eval_timeout_s=args.eval_timeout_s,
                                       skip_existing=args.skip_existing, args=args)
            except Exception as exc:  # noqa: BLE001
                res = {"instance_id": iid, "status": "orchestrator_crash",
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc(), "ended_at": _iso_now()}
                crash_dir = _safe_task_dir(per_task_root, iid)
                crash_dir.mkdir(parents=True, exist_ok=True)
                (crash_dir / "runner_metadata.json").write_text(
                    json.dumps(res, indent=2) + "\n", encoding="utf-8"
                )
            verdict = (res.get("eval_report") or {}).get("verdict", res.get("status", "?"))
            print(f"[{_iso_now()}] <- {iid} verdict={verdict} "
                  f"patch_bytes={res.get('patch_bytes')} elapsed={time.time()-t0:.1f}s", flush=True)
    finally:
        _stop_proxy(proxy_proc, proxy_log)

    ended_at = _iso_now()
    summary = _aggregate(per_task_root, dataset_out / "campaign_summary.json",
                         dataset_out / "predictions.jsonl", started_at, ended_at, args.model_name)
    print(f"=== [{ended_at}] DONE n={summary['instances_total']} "
          f"resolved_rate={summary.get('resolved_rate')} verdicts={summary['verdict_counts']} ===",
          flush=True)
    predictions_path = dataset_out / "predictions.jsonl"
    valid_ids: set[str] = set()
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prediction = json.loads(line)
            if str(prediction.get("model_patch") or "").strip():
                valid_ids.add(str(prediction.get("instance_id") or ""))
    missing_predictions = sorted(set(instance_ids) - valid_ids)
    if missing_predictions:
        print(
            f"ERROR requested tasks lack a nonempty prediction: {missing_predictions}",
            file=sys.stderr, flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
