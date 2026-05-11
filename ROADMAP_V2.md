# Self-Improving Agent — Roadmap v2.0

> Forward-looking improvement plan. The current implementation (Phases 1–11)
> is already strong: 332 tests, 92% coverage, full LoRA + MLX + RAG +
> registry stack, real-data pipeline, automated improvement cycle. This
> document covers the remaining gaps between "good infrastructure" and
> "the local model is the production AI for the OH platform."

## Context

What's already done is genuinely impressive:

- Provider abstraction with router (Groq + MLX, fallback, performance tracking)
- LoRA fine-tuning on Apple Silicon via `mlx-lm`
- ChromaDB-backed RAG with sentence-transformer embeddings
- Golden-test evaluation harness with promotion gating
- Model registry with lifecycle (training → canary → production → deprecated)
- Self-play conversation generator with Groq-as-judge filtering (≥7/10)
- Improvement cycle script that ties it all together
- 332 tests, CI on every push

The gaps are not infrastructural — they're "the trained model is hard to actually use," "evaluation is JSON not a dashboard," "routing intelligence is a keyword list," and "no production observability."

---

## Bugs / Honesty issues

### B1. CLI is stubbed end-to-end
`src/self_improving_agent/cli.py` defines `sia evaluate`, `sia train`, `sia
serve`, `sia status` — every one prints `"Command 'xxx' not yet implemented."`
and exits 1. The package ships a CLI entry point that does nothing.

**Fix:** wire each subcommand to its existing module:
- `sia status` → registry summary + last eval pass rate per version
- `sia evaluate <version>` → loads adapter, runs `EvaluationHarness`, prints
  report, writes JSON to `evaluation/reports/`
- `sia train` → invokes the existing improvement-cycle pipeline
- `sia serve` → starts the new inference API (see T1.1)

**Effort:** Small (½ day for status + evaluate; rest depends on T1.1).
**Files:** `src/self_improving_agent/cli.py` (rewrite), new
`src/self_improving_agent/commands/{status,evaluate,train,serve}.py`.

### B2. Router complexity assessment is a keyword list
`providers/router.py` "intelligent routing" is `any(marker in user_msg for
marker in COMPLEXITY_MARKERS)` — 14 hardcoded strings. README calls this
"complexity analysis." A `"why this fails"` message routes to local; a
`"draft a SOC2 policy"` routes to local. The label oversells the substance.

**Fix:** replace with a small heuristic that's at least defensible:
token count + question depth + topic-domain check. Document honestly that
SIA uses a heuristic, not a classifier; offer a classifier upgrade path.

**Effort:** Small (1 day).
**Files:** `providers/router.py`, `tests/test_router.py`.

### B3. CI runs unit tests but not model-quality tests
GitHub Actions runs pytest + ruff. There's no check that a model regression
fails CI. Today the registry's `is_promotion_ready()` is the only quality
gate — and only fires when the pipeline runs, not when code changes.

**Fix:** add a CI job that runs the evaluation harness against the current
production model on every push. Gate merges on no regression. Cache the
adapter to avoid re-downloading on each run.

**Effort:** Small (½ day, plus debugging CI-on-macOS or moving to a remote
inference path).
**Files:** `.github/workflows/ci.yml`.

---

## Tier 1 — Transformational

### T1.1. Inference HTTP server (FastAPI, OpenAI-compatible)
**Problem:** SIA trains models but has no serving mechanism. To use the
trained adapter from another OH service, you have to write code that loads
MLX, the adapter, and runs `generate` yourself. There's no `sia serve`.

**Plan:**
- FastAPI app exposing `/v1/chat/completions` (OpenAI-compatible).
- Loads the registry's current `production` model + adapter at startup.
- Streaming response via SSE (parallels multi_agent_planner's T2.3).
- Health/metrics endpoint exposing the router stats (`local_ratio`,
  `fallback_rate`, p50/p95 latency).
- Reload-on-promotion: when the registry's production version changes,
  the server hot-swaps the adapter without a restart.

**Effort:** Medium (3 days).
**Files:** new `src/self_improving_agent/serving/server.py`,
`serving/router_app.py`, `serving/models.py` (OpenAI request/response
schemas), `tests/test_serving.py`.

### T1.2. Continuous evaluation dashboard
**Problem:** Eval reports are JSON files in `evaluation/reports/`. To
answer "is the model getting better?" you have to grep JSON. Adding a
small dashboard makes the data actionable.

**Plan:**
- Lightweight web UI (Flask + Chart.js or Streamlit — Streamlit is simpler)
- Plots: pass rate over time per version, category scores, regression
  detection, latency distribution
- Reads everything from `evaluation/reports/*.json` + `models/registry.json`
- One-button "compare version A vs B" view
- Optional: live update when new reports land

**Effort:** Medium (2 days, Streamlit) or Larger (4 days, custom Flask).
**Recommend Streamlit** — same stack as your existing scanalyzer + fashion-item
deployments.

**Files:** new `src/self_improving_agent/dashboard/app.py`, supporting modules.

### T1.3. Multi-judge evaluation for self-play
**Problem:** `generate_conversations.py` uses a single Groq 70B call as the
judge. One judge has a viewpoint; biases creep into training data. Adding
two more judges (different model or different temperature) with voting
yields higher-quality filtered conversations.

**Plan:**
- New `JudgePanel` abstraction: takes a conversation, returns score + rationale
- Concrete judges: `GroqJudge` (current behavior), `LocalJudge` (use SIA's
  own current production model — interesting "model judges itself" signal),
  `EnsembleJudge` (median of N panels)
- `EnsembleJudge(judges=[groq, local, groq_low_temp])` becomes the default;
  conversation kept only if median score ≥ threshold
- Per-judge calibration: track which judges are stricter/laxer over time

**Effort:** Medium (2 days).
**Files:** new `src/self_improving_agent/training/judges.py`, refactor
`scripts/generate_conversations.py`, tests.

### T1.4. Reasoning-aware router (replaces B2's keyword list)
**Problem:** Once B2's heuristic lands, the natural next step is a real
classifier. Routing decisions are a high-leverage spot: a wrong "send to
local" wastes a request, a wrong "send to remote" wastes money.

**Plan:**
- Use the interaction log as a training signal: which requests got bad
  responses locally (low rubric scores, user thumbs-down) → those should
  route remote next time
- Train a tiny classifier (logistic regression on embeddings, or a 50M-param
  fine-tuned distilbert) to predict "needs remote model"
- The router consults the classifier as a soft signal; combine with B2's
  heuristic, fall back to it if classifier confidence is low
- Track classifier accuracy in production; surface in dashboard (T1.2)

**Effort:** Medium-Large (3–4 days).
**Files:** new `src/self_improving_agent/providers/complexity_classifier.py`,
update `providers/router.py`, tests.

---

## Tier 2 — High-impact additions

### T2.1. SSE streaming on the inference API
**Problem:** T1.1's first cut returns full responses. Real OH clients (the
companion app, the OH website) expect token-streaming.

**Plan:** SSE endpoint that streams tokens via the existing
`BaseModelProvider.generate_stream()`. OpenAI's spec format. Reuses the
event-emission pattern from multi_agent_planner.

**Effort:** Small (1 day after T1.1 lands).
**Files:** `serving/server.py`.

### T2.2. Active-learning feedback loop
**Problem:** When a real user gets a bad response, that's the highest-value
training signal — but there's no path from "bad response in prod" to "this
goes in the next training batch."

**Plan:**
- API endpoint `/feedback` accepting (interaction_id, rating, optional
  correction text)
- Feedback rows captured in a new `FeedbackStore` (JSON or SQLite)
- `DatasetBuilder` consults the feedback store and up-weights interactions
  with negative feedback (or excludes them entirely, your call) plus pairs
  the corrections as preference data
- Dashboard view: feedback volume per version, regression markers

**Effort:** Medium (2 days).
**Files:** new `src/self_improving_agent/feedback/store.py`, update
`training/dataset_builder.py`, `serving/server.py`, tests.

### T2.3. Distillation pipeline (Groq → local)
**Problem:** Groq 70B answers questions extremely well. SIA's local 8B
model is the deployment target. There's no end-to-end script that takes a
question set, fetches Groq's responses, and trains the local model to
mimic them. The pieces exist (Groq provider, dataset builder, training
script) but no orchestration.

**Plan:**
- New script `scripts/distill_from_groq.py`:
  1. Sample N user questions (from interaction log or a question pool)
  2. Get Groq 70B's responses for each
  3. Filter for high-quality responses (LLM-as-judge or static rubrics)
  4. Format as SFT pairs
  5. Hand off to existing training pipeline
- Optional: also store the rationale ("why is this a good answer") for
  later use in preference training

**Effort:** Small-Medium (1–2 days; mostly orchestration of existing pieces).
**Files:** new `scripts/distill_from_groq.py`, tests.

### T2.4. LoRA adapter merging
**Problem:** Each training run produces a separate adapter. Today there's
no way to combine an adapter trained on, say, "general OH knowledge" with
one trained on "Bianca-specific brand voice."

**Plan:**
- New `AdapterMerger` that takes N adapter paths and weights, produces a
  single merged adapter (via TIES-merging or task-arithmetic, both
  well-supported in MLX)
- CLI: `sia merge --base v1.0 --adapter brand:0.7 --adapter persona:0.3
  --out v1.0-merged`
- Eval harness can verify the merged adapter doesn't regress on either
  source's evaluation set

**Effort:** Medium (2 days).
**Files:** new `src/self_improving_agent/training/merging.py`, CLI hook,
tests.

### T2.5. Multi-base-model evaluation
**Problem:** Everything is locked to Llama-3.1-8B. The infrastructure is
generic but the evaluation pipeline hardcodes it. Comparing Qwen2-7B or
DeepSeek-Coder-6.7B on the same eval suite should be a flag, not a fork.

**Plan:**
- `EvaluationHarness` already takes a `generate_fn`. Make the registry +
  pipeline configurable so a non-Llama base can be registered, trained
  (or downloaded with a pre-existing adapter), and evaluated.
- Add `base_model` to the EvaluationReport for clarity.
- Dashboard view: pass-rate-per-base-model.

**Effort:** Medium (2 days).
**Files:** `training/registry.py`, `pipeline/self_improve.py`,
`evaluation/models.py`, tests.

---

## Tier 3 — Polish & ops

### T3.1. Cost tracking
**Problem:** Each Groq API call costs money on Dev tier. There's no
per-script or per-run accounting. The multi-agent-planner has a
`CostTracker` we built last week — port the design.

**Plan:** thread-safe accumulator with per-(provider, model) pricing,
record_usage() from each provider, render_summary() at end of every
script. Same pattern as `multi_agent_planner/core/cost_tracker.py`.

**Effort:** Small (1 day).
**Files:** new `src/self_improving_agent/observability/cost_tracker.py`,
hook into both providers, tests.

### T3.2. Reproducibility seeds
**Problem:** None of the training scripts pin random seeds. Two consecutive
runs of `train_oh_model_v11.py` produce different adapters. That's a
reproducibility problem for the eval harness *and* for blaming a regression
on a specific data change.

**Plan:**
- `--seed` flag on every training script, defaulting to a fixed value
- Plumb through to mlx-lm, numpy, random, and tokenizer batching
- Document deterministic-mode caveats (gradient checkpointing + half
  precision still leak non-determinism)
- Stamp seed into `ModelVersion.metadata` so reports trace back

**Effort:** Small (½ day).
**Files:** every script in `scripts/`, `training/finetune.py`.

### T3.3. Script utility consolidation
**Problem:** `scripts/` has 9 files totaling ~70KB. They share patterns —
argparse setup, env loading, Groq client init, progress bars — that get
copy-pasted. A small `scripts/_lib.py` would dedupe.

**Plan:** extract:
- `load_env_or_die()` — Groq key check, .env load
- `init_logging(verbose)` — consistent log format
- `make_progress(total, label)` — tqdm wrapper with auto-detect
- `resolve_paths(args)` — handle relative vs absolute path resolution

**Effort:** Small (½ day, mostly mechanical refactor).
**Files:** new `scripts/_lib.py`, every script.

### T3.4. LLM-as-judge rubric mode in the eval harness
**Problem:** Golden tests use static keyword matching for scoring. Some
behaviors (tone, helpfulness, factuality) are best judged by another LLM.

**Plan:** add `judge` rubric type to the golden YAML schema. When a rubric
specifies `judge: groq`, the harness calls Groq with a "score this
response 0-10 against this criterion" prompt. Standard rubrics stay
unchanged.

**Effort:** Small (1 day).
**Files:** `evaluation/harness.py`, sample rubric in golden_tests/.

### T3.5. Monitoring / observability
**Problem:** Once the serving API is up, you need to know: which version
is running, what its p50 latency is, what the fallback rate looks like,
which questions are getting low-rated responses. Currently there's no
metrics endpoint and no Grafana wiring.

**Plan:**
- Prometheus-format `/metrics` endpoint on the serving API
- Expose: `sia_requests_total{provider,model}`,
  `sia_latency_seconds{provider}`, `sia_fallback_total{from,to}`,
  `sia_version_info{version,status}`
- Sample dashboard JSON for Grafana so the user can spin one up

**Effort:** Small-Medium (1 day).
**Files:** `serving/metrics.py`, sample `monitoring/grafana.json`.

---

## Recommended phasing

### Sprint 1 (~2 days): "Trust + visibility"
- B1 (CLI implementation)
- B3 (CI eval gating)
- T3.1 (cost tracking)
- T3.2 (reproducibility seeds)

After: every claim the README makes is true, every run is reproducible,
every API call is accounted for.

### Sprint 2 (~3-4 days): "The model is usable"
- T1.1 (FastAPI inference server)
- T2.1 (SSE streaming)
- T1.2 (Streamlit dashboard)

After: SIA's trained adapters are addressable over HTTP and the eval data
is visible.

### Sprint 3 (~3-4 days): "Smarter signals"
- B2 (router heuristic upgrade)
- T1.3 (multi-judge ensemble)
- T2.3 (distillation pipeline)
- T3.4 (LLM-as-judge rubrics)

After: routing is honest and the training data going in is higher quality.

### Sprint 4 (~3 days): "Operational"
- T2.2 (feedback loop)
- T2.4 (adapter merging)
- T2.5 (multi-base-model eval)
- T3.3 (script utilities)
- T3.5 (Prometheus metrics)

After: SIA has a closed feedback loop from production back to training,
multiple bases are comparable on the same harness, and there's a real
monitoring story.

### Sprint 5 (~3-4 days): "Reasoning-aware"
- T1.4 (classifier-based router; depends on enough interaction-log data
  having accumulated)

After: routing decisions are made by a model trained on actual outcomes,
not a list of strings.

---

## Total scope

- **Bugs:** 1.5 days
- **Tier 1:** ~10 days
- **Tier 2:** ~9 days
- **Tier 3:** ~3 days

≈ **4-5 weeks** of focused work to land everything in this document.

## Cross-repo opportunities

Three pieces deliberately mirror what we just built in `multi_agent_planner`:

- **T3.1 cost tracker** — same design as MAP's; could even share the module
- **T1.1 SSE streaming** — same event-bus pattern
- **T1.2 eval dashboard** — the dashboard could read both projects' eval
  reports for a unified "OH AI health" view

If SIA becomes pip-installable (or a workspace dep), MAP could import
SIA's `EvaluationHarness` directly instead of vendoring an equivalent.
That's the original ROADMAP_V2 intent — "treat SIA as a library MAP
depends on."
