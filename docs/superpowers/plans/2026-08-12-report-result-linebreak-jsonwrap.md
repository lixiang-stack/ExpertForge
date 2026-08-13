# Report HTML Result Line-Break + JSON Wrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two report.html rendering issues: the result (总结) stage gets a line break between its header and content, and long JSON decision values wrap instead of overflowing.

**Architecture:** Confined to the renderer in `agent/observability/report.py` and `tests/test_report.py`. The result branch of `_stage_li` changes from a single inline line to a header + nested content `<li class="result">`. A CSS rule makes `pre.json` wrap long lines (`white-space:pre-wrap; overflow-wrap:anywhere`). No data-model changes, no value truncation.

**Tech Stack:** Python 3.12 (repo standard), stdlib, inline HTML/CSS.

## Global Constraints

- **Token-only:** no monetary cost anywhere.
- **Never raise into business code:** rendering degrades with `warnings.warn` + non-zero exit on write failure (unchanged).
- **No schema change, no new event fields, no recording-layer changes.**
- **HTML self-contained:** single file, inline CSS, no CDN/JS. All user content `html.escape`'d (unchanged).
- **No truncation of JSON values** — long values wrap via CSS only.
- The `.result` CSS color class already exists; reuse it for the result content line.
- **Amended (human ruling):** stage headers render as the plain phase/title string (e.g. `classification`, `route`, `result` — no `阶段`/`结果`/`总结` suffixes); decision JSON `<pre>` uses `class='json'`; no `结构化输出` prefix on decision rows. These were pre-existing uncommitted edits the implementer preserved; the user accepted them and amended this plan to match.

---
## File Structure

| File | Responsibility |
|------|----------------|
| `agent/observability/report.py` (modify) | Result stage: header + content on separate lines; add `pre.json` wrap CSS rule. |
| `tests/test_report.py` (modify) | Add/adjust assertions for the two-line result structure and the CSS wrap rule. |

---
### Task 1: result line-break + JSON wrap CSS

**Files:**
- Modify: `agent/observability/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `build_html_report(events, *, default_collapsed=True) -> str` (existing), `Stage`, `summarize_traces` (existing). The result branch of `_stage_li` uses `summary.reject`, `summary.in_tokens`, `summary.out_tokens`, `summary.total_latency_ms`, and the result step's `answer_len`.
- Produces: updated `build_html_report` with the result stage on two lines and `pre.json` wrapping.

- [ ] **Step 1: Update the tests** — in `tests/test_report.py`, add a test for the two-line result structure and the CSS wrap rule:

```python
def test_html_result_header_on_own_line():
    html = build_html_report(_events())
    assert "<b>result</b>" in html
    # the content follows in a nested block-level <ul>, so it renders on its own
    # line below the header (block elements stack vertically in the browser)
    assert "<b>result</b><ul><li class=\"result\">" in html
    assert "answer_len=50 reject=False, total in=30 out=15 tokens, time=0.3s" in html


def test_html_pre_json_wraps_long_lines():
    html = build_html_report(_events())
    assert "white-space:pre-wrap" in html
    assert "overflow-wrap:anywhere" in html
```

Keep all existing tests unchanged. Note the existing `test_html_has_time_labels` asserts `time=0.3s total` (meta strip) and `time=0.3s` (card summary) — the result content line `time=0.3s` also appears; the two-line change must keep `time=0.3s` present in the html. `test_html_has_summary_and_labels` and `test_html_details_collapsed_by_default` are unaffected.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report.py -k "result or pre_json" -q`
Expected: FAIL — the result stage is still one inline line and `white-space:pre-wrap` is absent.

- [ ] **Step 3: Implement** — in `agent/observability/report.py`:

In `_stage_li`, replace the `result` branch (lines ~36-41):

```python
    if title == "result":
        result_step = next((s for s in stage.steps if s.kind == "result"), None)
        answer_len = (result_step.detail or {}).get("answer_len") if result_step else ""
        content = (f'<li class="result">answer_len={answer_len} reject={summary.reject}, '
                   f'total in={summary.in_tokens} out={summary.out_tokens} tokens, '
                   f'time={summary.total_latency_ms / 1000:.1f}s</li>')
        return f'<li class="stage">{header}<ul>{content}</ul></li>'
```

This emits the header, then a nested `<ul>` with one `<li class="result">` carrying the content — so the header and content are on separate lines.

Add the CSS rule. In the `<style>` block (line ~148-153), after the `pre.json`/`.decision` line, add:

```css
pre.json{white-space:pre-wrap;overflow-wrap:anywhere;margin:.2rem 0}
```

(The `.json` class is already emitted by `_decision_json_html`; if a `pre.json` rule already exists in the style block, extend it rather than duplicating — check first.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report.py tests/test_report_data.py -q`
Expected: PASS (report + data tests all green)

- [ ] **Step 5: Smoke-test against real data**

Run: `uv run python -m agent.observability report --data-dir .observability`
Expected: prints `Report written to .observability/report.html`. Open the file: each trace's `result` stage shows `result` on its own line with the content indented below; long `reason` values in the JSON `<pre>` blocks wrap within the block instead of overflowing horizontally.

- [ ] **Step 6: Full suite**

Run: `uv run pytest -q`
Expected: all green (144 passed, 4 skipped baseline + new tests)

- [ ] **Step 7: Commit**

```bash
git add agent/observability/report.py tests/test_report.py
git commit -m "fix: result stage line break; JSON pre wraps long values"
```

---
## Self-Review

**1. Spec coverage:**
- Result header on own line + content line → Task 1 (`_stage_li` result branch emits header + nested `<ul><li class="result">`).
- JSON `<pre>` wraps long values via CSS, no truncation → Task 1 (`pre.json{white-space:pre-wrap;overflow-wrap:anywhere;margin:.2rem 0}`).
- Tests → `test_html_result_header_on_own_line`, `test_html_pre_json_wraps_long_lines`.

**2. Placeholder scan:** no TBD/TODO; every step has real code and runnable commands.

**3. Type consistency:** the result branch still uses the same fields (`summary.reject/in_tokens/out_tokens/total_latency_ms`, result step `answer_len`) — only the HTML structure changes. `_decision_json_html` unchanged (still emits `<pre class='json'>`). The `header` variable is defined at the top of `_stage_li` (line 35) before the result branch, so it's available. The CSS class `.json` matches the class emitted by `_decision_json_html`. The two-line test asserts the exact nested structure the new code emits.