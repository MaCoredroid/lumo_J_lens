# Chat Template Provenance

The runtime chat template is intentionally fetched rather than committed to this
repository. `scripts/fetch_chat_template.sh` downloads the exact file from:

```text
repository: MaCoredroid/Lumo_FlyWheel
commit: 9fd1b40287748c8b6b8a9075fc383f454a30b0e0
path: docker/chat_templates/qwen3-openai-codex.jinja
SHA-256: c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da
```

Repository history records this lineage:

1. Lumo_FlyWheel commit `542c8e1118c8` replaced the earlier template with
   `qwen3.5-enhanced.jinja` from
   `allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix`, then added support for
   developer-role messages.
2. Lumo_FlyWheel commit `9fd1b4028774` added the interleaved-thinking behavior
   used by the July 8 serving profile.

The baseline project's README labels it `MIT`, but it has no separate LICENSE
file or detailed copyright notice. The final fetched template also has its own
reference header. It is therefore excluded from this repository's MIT grant and
from Git tracking. Review the named upstream sources and establish the terms
appropriate to your use before redistributing the fetched file.
