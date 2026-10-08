# AGENTS.md

Dify model-provider plugin for [ZenMux](https://zenmux.ai). Upstream: `ZenMux/dify-plugin` (maintainers merge,
tag the release and open the Marketplace PR in `langgenius/dify-plugins`). Day-to-day maintenance happens on a fork.

## Layout

| Path | What |
|---|---|
| `provider/zenmux.yaml` | Provider schema: credentials (`api_key`, `region` = `global` → zenmux.ai, `cn` → zenmux.dev), model types |
| `models/_common.py` | Region → base URL, real-cost backfill (`usage.cost`), local reasoning stripping |
| `models/llm/zenmux.py` | Routes predefined models by id prefix: `anthropic/*` → Anthropic native, `google/gemini-*` → Vertex, rest → OpenAI-compatible. Custom (non-predefined) models always use OpenAI-compatible |
| `models/llm/{openai,anthropic_llm,google}.py` | One class per ZenMux protocol |
| `models/text_embedding/`, `models/rerank/` | Embeddings (OpenAI-compatible), rerank (ZenMux `input`/`parameters` body, raw scores) |
| `models/**/*.yaml`, `models/llm/_anthropic_thinking.json` | **Generated** — never hand-edit; change `catalog/overrides.yaml` and re-run sync |
| `catalog/` | `snapshot.json` (normalized live catalog), `probes.json` (live probe results), `overrides.yaml` (manual knowledge with reasons + dates) |
| `scripts/zenmux_models.py` | `fetch` / `probe` / `sync` / `check` |
| `scripts/package.sh` | Builds the `.difypkg` from a clean `git ls-files` copy |
| `tests/unit` | Offline; loads the plugin exactly like the daemon does |
| `tests/live` | Real calls through the plugin classes (needs a key, a few cents) |
| `e2e/run_matrix.py` | Installs the package into official Dify stacks (1.6.0 / 1.13.1 / 1.14.2 / 1.17.1) and chats through the console API |

## Setup

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt pytest pyyaml
```

Packaging CLI: download `dify-plugin-<os>-<arch>` from `langgenius/dify-plugin-daemon` releases and verify the
sha256 digest shown on the release page; point `DIFY_PLUGIN_CLI` at it. Unsigned binaries may need
`codesign -s - <file>` on macOS.

## Release workflow (model sync → PR)

1. **Branch** from up-to-date `origin/main`: `feat/sync-models-<version>`.
2. **Catalog**: `.venv/bin/python scripts/zenmux_models.py fetch`
3. **Probe** new/changed models (PAYG key recommended; < $1):
   `ZENMUX_PROBE_API_KEY=... .venv/bin/python scripts/zenmux_models.py probe` (`--all` to refresh everything,
   `--only <regex>` for a subset). Probes detect retired models, tool calling, `reasoning_effort` values,
   Claude temperature/adaptive thinking and how thinking is turned off, embedding batching, and whether embedding/rerank models that the
   catalog lists with image input really use the image (coloured squares must match their captions).
   Re-probe `--only 'embedding|rerank'` on every release: ZenMux fixes or breaks image support without notice.
4. **Sync**: `.venv/bin/python scripts/zenmux_models.py sync > /tmp/sync_report.md`. Read the report:
   - *Excluded → no pricing*: verify the real rate via `GET /api/v1/management/generation?id=<id>`
     (`ratingDetails[].rate` is USD per 1M tokens) and add it under `manual` with a `verified` date — or leave it out.
   - Suspicious capability/param changes → fix via `overrides.yaml` `models:` entries, with a reason.
   - *Claude thinking off-switch*: compare with Anthropic's per-model table
     (platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting). `always_on` models get no switch.
   - *Image input listed but not verified*: these ship text-only (no `vision`, so Dify won't send them images).
     Only force it with `features_add: [vision]` after checking real retrieval quality yourself.
5. **Code changes** (if any), then `.venv/bin/python -m pytest tests/unit -q` — must be green
   (includes `sync --check`, so YAMLs cannot drift from `catalog/`).
6. **Live**: `ZENMUX_LIVE_API_KEY=... .venv/bin/python -m pytest tests/live -q` (add `ZENMUX_LIVE_REGION=cn`
   to cover zenmux.dev).
7. **Version**: bump `manifest.yaml` `version`; update README "Version".
8. **Package**: `DIFY_PLUGIN_CLI=... scripts/package.sh dist` — check the printed file list has no stray files.
   Then run the Marketplace pre-check that `langgenius/dify-plugins` PRs go through
   (`gh repo clone langgenius/dify-marketplace-toolkit`, needs `yq`):
   `python3 validator/validate-difypkg.py dist/zenmux-dify-plugin_<v>.difypkg --output-dir /tmp/mkt-report` —
   all blocking checks must PASS (it rejects e.g. `.pem` files and requires `repo`/`contact` in the manifest).
9. **Dify matrix**: `ZENMUX_E2E_API_KEY=... .venv/bin/python e2e/run_matrix.py --pkg dist/zenmux-dify-plugin_<v>.difypkg`.
   Needs Docker with ~3 GB free RAM; runs versions one at a time and tears stacks down (`--keep` to inspect).
10. **PR** to `ZenMux/dify-plugin` from the fork; paste the sync report and the test/matrix results into the body.
    After merge, upstream maintainers publish the release and the Marketplace PR. Follow upstream conventions:
    English Conventional Commits (`feat:`/`fix:`/`chore:`), one logical change per commit, PR body with
    Summary / Changes / Test plan (see PR #7). The Marketplace PR template also asks for a Dify Cloud test.

## Rules of thumb

- Billing: Dify shows `usage.total_price`. On OpenAI-compatible and Anthropic routes it is ZenMux's `usage.cost`,
  which equals the bill's `originAmount` (tiers and cache applied, promotional discounts NOT applied; for
  discounted models `billAmount` is lower). Vertex (Gemini) and embeddings fall back to YAML base-tier prices.
- ZenMux ignores `reasoning.exclude`; "hide thought process" is enforced in `strip_reasoning`.
- Claude thinking follows Anthropic's per-model table: newer models think by default, so "off" is sent explicitly
  (`disabled`, or `between_tools` on Sonnet 5.5); always-on models (Opus 5.5, Fable) cannot turn it off.
  Check provider docs before calling something a ZenMux bug; ZenMux's own docs lag behind new models.
- Free (`-free`) and zero-priced models are excluded on purpose.
- Keep `requirements.txt` pinned to exact versions that passed the Dify matrix; prefer releases older than 7 days.
- Secrets: keys only via environment variables; `.env` is ignored by git and by packaging.
