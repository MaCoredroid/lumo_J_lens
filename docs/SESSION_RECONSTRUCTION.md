# Source Session Reconstruction

## Authoritative Session

The source was the most recent Claude Code session that implemented the 27B
teacher swap:

The transcript was read from the local Claude project-history JSONL store. Its
machine username, full session UUID, and unrelated transcript content are not
published here.

The focused request begins at JSONL line 15,726, timestamp
`2026-07-08T16:43:50.750Z` (09:43:50 PDT). The user requested the official
Qwen3.6-27B NVFP4 model, native MTP, flywheel prefix caching, capped DDR5 KV
offload, and Qwen Code for SWE-bench Verified rather than Codex.

The final stack summary appears near line 16,015 at `20:16:42Z`. Follow-up work
later fixed context overflows and stale transient scopes.

## Timeline

| PDT | Work |
|---|---|
| Jul 8 09:43 | 27B NVFP4 + MTP + APC request |
| 09:47-10:16 | Official model inspection/download; flywheel sync |
| 10:17-11:05 | Boot and memory/offload grid |
| 11:16-11:52 | Format, grounding, and live SWE gates |
| 11:50-12:29 | Official configuration audit; Qwen Code 0.19.4 adoption |
| 13:15 | Final thinking-envelope relaunch |
| 14:03-14:07 | Context-limit clamp and retry |
| 14:45 | Stale-scope reset hardening |

## Source Commits

| Commit | Meaning |
|---|---|
| `c4b82da` | Stage the 27B launcher |
| `0eab16d` | Freeze boot fixes and seq2/off0 profile |
| `423cdd3` | Official envelope/config audit |
| `69b39c3` | Initial 27B teacher activation |
| `bf7c0cf` | Adopt Qwen Code 0.19.4 |
| `69ce1ea` | Final thinking-envelope stack and relaunch |
| `a040319` | Add the context-fit clamp/retry |
| `d7805e7` | Reset stale systemd unit before boot |

The flywheel fork merge was `b32f08d4` from upstream ref `78af0be4`. The chat
template includes the interleaved-thinking changes from `9fd1b402`.

## Historical Evidence

The structured frozen configuration, gate verdict, and teacher-swap report were
read from the source repository's ignored run-artifact tree. Local absolute
paths are omitted from this public reconstruction.

The historical gate established the server, MTP, cache, tool format, grounding,
and official-container path. Its two Verified tasks used Qwen Code 0.19.2 and
the earlier temperature-0.6 envelope. The July 15 publication-certified run in
this repository provides exact validation of 0.19.4 plus the final
temperature-1.0 thinking envelope and shell-only security boundary.

No Claude authentication data, GitHub token, SSH key, sudo password, or other
secret from the local transcripts is included in this repository.
