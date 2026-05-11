# CI secrets

GitHub Actions in this repo reads two secrets. They live in
**Settings → Secrets and variables → Actions → New repository secret** at:
https://github.com/oh-systems/self-improving-agent/settings/secrets/actions

| Secret | Required by | What it does |
|---|---|---|
| `GROQ_API_KEY` | `eval-remote` job (`.github/workflows/ci.yml`) | When set, the workflow runs the golden-test harness against the remote Groq provider on every push to `main`. When unset, that job is a no-op (logs "skipping" and exits 0). Soft-gated with `continue-on-error: true` until a stable baseline is established. |
| `CODECOV_TOKEN` | (future) coverage upload | Optional. If/when we wire `codecov/codecov-action`, this is the upload token. |

## Promoting `eval-remote` to a hard gate

Once the baseline is stable, edit `.github/workflows/ci.yml` and:

1. Remove `continue-on-error: true` from the `eval-remote` job.
2. Add a check that the pass rate is at or above a chosen threshold.
3. Optional: cache the golden tests so the job runs in <60s.

The first two changes turn pushes to `main` into a contract: a regression
in remote inference quality fails CI.

## Local equivalent

```bash
export GROQ_API_KEY=gsk_...
python -m self_improving_agent.cli evaluate --remote --no-write
```

Same command CI runs.
