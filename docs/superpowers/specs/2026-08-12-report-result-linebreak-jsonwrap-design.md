# Report HTML Result Line-Break + JSON Wrap

**Date:** 2026-08-12
**Status:** Approved design (pending implementation plan)

## Problem Statement

Two rendering issues in the generated `.observability/report.html`:

1. **The result (总结) stage renders header and content on one line.** The trace timeline's result stage emits `<b>result</b> answer_len=... reject=..., total in=... out=... tokens, time=...s` as a single line. The header should be on its own line with the content below it.

2. **Long JSON decision values overflow horizontally.** Decision `<pre>` blocks (classification/route JSON) contain long string values (e.g. the `reason`/`reject_reason` sentences), which render as very long single lines that overflow the `<pre>` block instead of wrapping.

## Design

Scope is confined to `agent/observability/report.py` rendering and `tests/test_report.py`. No data-model, schema, or recording-layer changes.

### 1. Result stage: header line + content line

In `_stage_li`, the `result` branch currently returns a single inline line. Change it to emit the header, then the content on the next line, wrapped like other stages' bodies:

```html
<li class="stage"><b>result</b>
  answer_len=3202 reject=False, total in=536 out=1568 tokens, time=15.5s</li>
```

Implementation: the result branch emits `<b>result</b>` followed by a nested `<ul>` containing one `<li class="result">` with the content string (`answer_len=... reject=..., total in=... out=... tokens, time=...s`). This matches the structure other stages use for their bodies and reuses the existing `.result` CSS color class.

### 2. JSON `<pre>` long-value wrapping (CSS only, no truncation)

Keep full values. Add a CSS rule so long lines wrap instead of overflowing:

```css
pre.json{white-space:pre-wrap;overflow-wrap:anywhere;margin:.2rem 0}
```

- `white-space: pre-wrap` preserves the JSON's newlines/indentation while wrapping long lines.
- `overflow-wrap: anywhere` breaks otherwise-unbreakable long strings (the reason sentences).
- No Python-side value truncation.

## Data Flow

Unchanged: `read_events` → `report_data` → `report.py` renderer → `report.html`. Only the result-stage HTML structure and one CSS rule change.

## Error Handling

None new — the changes are pure rendering/CSS; no new code paths that can raise.

## Testing

- **`tests/test_report.py`**:
  - Update/add an assertion that the result stage renders the header on its own line followed by the content (`answer_len=50 reject=False, total in=...` present; header `总结`-equivalent `result` present). The existing `test_html_has_summary_and_labels` and `test_html_has_time_labels` already assert content substrings (`time=0.3s`, `total in=...`) — verify they still pass with the two-line structure.
  - Add an assertion that the `<pre class='json'>` element or the CSS includes `white-space:pre-wrap` / `overflow-wrap:anywhere` (e.g. assert `"white-space:pre-wrap"` in the html).
- Full suite stays green.

## Out of Scope

- Any truncation of JSON values.
- Any data-model or recording-layer change.
- Changing the trace card `<summary>` header, the top summary strip, or the charts.