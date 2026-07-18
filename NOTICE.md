# Provenance Notice

This repository integrates, but does not vendor the distributions of, these
upstream projects:

- vLLM: `vllm-project/vllm`
- Qwen Code: `QwenLM/qwen-code`
- SWE-bench: `princeton-nlp/SWE-bench`
- Qwen3.6-27B-NVFP4 checkpoint: `nvidia/Qwen3.6-27B-NVFP4`
- SymPy: `sympy/sympy`
- Django: `django/django`
- Anthropic Jacobian Lens reference implementation: `anthropics/jacobian-lens`
- Neuronpedia pre-fitted lens collection: `neuronpedia/jacobian-lens`
- Qwen3.6 Apple/MLX architecture reference: `WeZZard/jlens-qwen36`

The chat template is not vendored. `scripts/fetch_chat_template.sh` retrieves it
from the public `MaCoredroid/Lumo_FlyWheel` repository at immutable commit
`9fd1b40287748c8b6b8a9075fc383f454a30b0e0` and verifies SHA-256
`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`.
Its baseline came from `allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix`,
whose README designates the project as MIT, and was then modified for developer
messages and interleaved thinking. The fetched file is external and is not
covered by this repository's MIT license. See
`docs/CHAT_TEMPLATE_PROVENANCE.md` before redistributing it.

Model weights, official task images, full dataset distributions, package
caches, and credentials are not committed here. They retain their upstream
licenses and terms. One exact SWE-bench Verified dataset row for
`django__django-13297` is committed as provenance, together with its captured
Qwen Code trajectory, generated patch, official-score records, selected full
Django source/test file contents, and shorter excerpts in the conversation.
The upstream Qwen Code Apache-2.0 and SWE-bench MIT license texts are reproduced
in `validation/QWEN_CODE_LICENSE` and `validation/SWE_BENCH_LICENSE`,
respectively. The Django portions are covered by the BSD license reproduced in
`validation/DJANGO_LICENSE`.
`validation/sympy__sympy-13480.patch` includes a small SymPy source diff as
exact run evidence and is covered by the SymPy BSD license reproduced in
`validation/SYMPY_LICENSE`. The repository's original scripts and documentation
are MIT licensed.

The reviewed C0M/C1 prompt, runner-report, and request-capture evidence under
`validation/jlens-swe-multitask-*` and
`validation/swe-multitask-c1-capture-*`, plus the full single-task records under
`validation/swe-multistage-django-13297` and the derived multistage replay,
consists of exact experiment evidence rather than a repository or full-dataset
distribution. It contains rendered Qwen Code system text, public SWE-bench task
inputs, the one dataset row described above, selected full Django source/test
file contents, and shorter tool-output excerpts from the named task
repositories, plus non-secret local paths and run identifiers needed for
byte-exact prompt provenance. Those embedded upstream portions retain their
respective upstream licenses and are not relicensed under this repository's MIT
license. No credentials or package, model, task-image, or full repository
distribution is included.

The Neuronpedia lens is downloaded on demand and is not committed. The
Anthropic and WeZZard repositories are Apache-2.0 references; no MLX/Metal
implementation is vendored or executed by this project.
