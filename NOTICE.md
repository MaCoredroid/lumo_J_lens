# Provenance Notice

This repository integrates, but does not vendor the distributions of, these
upstream projects:

- vLLM: `vllm-project/vllm`
- Qwen Code: `QwenLM/qwen-code`
- SWE-bench: `princeton-nlp/SWE-bench`
- Qwen3.6-27B-NVFP4 checkpoint: `nvidia/Qwen3.6-27B-NVFP4`
- SymPy: `sympy/sympy`

The chat template is not vendored. `scripts/fetch_chat_template.sh` retrieves it
from the public `MaCoredroid/Lumo_FlyWheel` repository at immutable commit
`9fd1b40287748c8b6b8a9075fc383f454a30b0e0` and verifies SHA-256
`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`.
Its baseline came from `allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix`,
whose README designates the project as MIT, and was then modified for developer
messages and interleaved thinking. The fetched file is external and is not
covered by this repository's MIT license. See
`docs/CHAT_TEMPLATE_PROVENANCE.md` before redistributing it.

Model weights, official task images, full datasets, package caches, and
credentials are not committed here. They retain their upstream licenses and
terms. `validation/sympy__sympy-13480.patch` includes a small SymPy source diff
as exact run evidence and is covered by the SymPy BSD license reproduced in
`validation/SYMPY_LICENSE`. The repository's original scripts and documentation
are MIT licensed.
