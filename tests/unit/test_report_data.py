from agent.observability.report_data import (
    ModelStat,
    TraceSummary,
    model_stats,
    summarize_traces,
    total_stats,
)


def _events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q1", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m1",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.direct", "model": "m2",
         "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
         "latency_ms": 200, "status": "ok", "ts": 20},
        {"type": "trace_end", "trace_id": "a", "answer_len": 50, "total_llm_calls": 2,
         "total_tokens": 45, "total_latency_ms": 300.0, "reject": False,
         "phase": "trace", "ts": 300},
        {"type": "llm_call", "trace_id": "b", "phase": "classification", "model": "m1",
         "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
         "latency_ms": 50, "status": "error", "error": "boom", "ts": 5},
    ]


def test_summarize_traces_aggregates():
    rows = summarize_traces(_events())
    assert len(rows) == 2
    a = rows[0]
    assert a.trace_id == "a"
    assert a.question == "q1"
    assert a.domain == "sw"
    assert a.in_tokens == 30
    assert a.out_tokens == 15
    assert a.total_tokens == 45
    assert a.llm_calls == 2
    assert a.total_latency_ms == 300.0
    assert a.reject is False
    assert a.has_error is False
    b = rows[1]
    assert b.has_error is True
    assert b.in_tokens == 0
    assert b.out_tokens == 0
    assert b.total_latency_ms == 50.0


def test_summarize_missing_usage_counts_zero():
    rows = summarize_traces([
        {"type": "llm_call", "trace_id": "x", "phase": "route", "model": "m1",
         "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
         "latency_ms": 10, "status": "ok"},
    ])
    assert rows[0].in_tokens == 0
    assert rows[0].out_tokens == 0
    assert rows[0].llm_calls == 1


def test_total_stats():
    st = total_stats(_events())
    assert st["traces"] == 2
    assert st["llm_calls"] == 3
    assert st["in_tokens"] == 30
    assert st["out_tokens"] == 15
    assert st["total_tokens"] == 45
    assert st["total_latency_ms"] == 350.0
    assert st["has_error"] is True


def test_model_stats_sorted_and_aggregated():
    ms = model_stats(_events())
    assert [m.model for m in ms] == ["m2", "m1"]  # by total tokens desc (30, 15)
    assert ms[0].calls == 1
    assert ms[0].in_tokens == 20
    assert ms[0].out_tokens == 10
    assert ms[1].calls == 2
    assert ms[1].in_tokens == 10


def test_summary_as_dict():
    s = summarize_traces(_events())[0].as_dict()
    assert s["trace_id"] == "a"
    assert s["in_tokens"] == 30


def test_model_stat_is_dataclass():
    assert isinstance(model_stats(_events())[0], ModelStat)
    assert isinstance(summarize_traces(_events())[0], TraceSummary)


def _decision_events():
    return [
        {"type": "trace_start", "trace_id": "a", "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 10,
         "data": {"intent": "question", "complexity": "low", "reason": "simple"}},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m1",
         "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
         "latency_ms": 100, "status": "ok", "ts": 20},
        {"type": "decision", "trace_id": "a", "phase": "route", "ts": 30,
         "data": {"strategy": "s1", "orchestrate": True}},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.planner", "ts": 35,
         "data": {"tasks": [{"title": "t1", "instruction": "i1"}]}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m2",
         "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
         "latency_ms": 50, "status": "ok", "ts": 40},
        {"type": "trace_end", "trace_id": "a", "answer_len": 30, "reject": False,
         "total_latency_ms": 150.0, "phase": "trace", "ts": 50},
    ]


def test_build_timeline_orders_by_ts():
    tl = build_timeline(_decision_events())
    assert list(tl) == ["a"]
    steps = tl["a"]
    assert [s.kind for s in steps] == ["decision", "llm_call", "decision", "decision",
                                       "llm_call", "result"]
    assert [s.phase for s in steps] == ["classification", "classification", "route",
                                        "orchestration.planner", "orchestration.worker.1", "trace"]
    assert steps[0].detail["type"] == "classification"
    assert steps[0].detail["data"]["intent"] == "question"
    assert steps[2].detail["type"] == "route"
    assert steps[3].detail["data"]["tasks"][0]["title"] == "t1"
    assert steps[4].detail["in_tokens"] == 2
    assert steps[5].detail["reject"] is False


def test_build_timeline_worker_decision_type():
    tl = build_timeline([{"type": "decision", "trace_id": "a", "phase": "orchestration.worker.2",
                          "ts": 1, "data": {"task": "fix bug"}}])
    steps = tl["a"]
    assert steps[0].kind == "decision"
    assert steps[0].detail["type"] == "worker"
    assert steps[0].detail["data"]["task"] == "fix bug"


def test_build_timeline_ignores_unknown_and_missing_trace():
    assert build_timeline([{"type": "wat", "trace_id": "x", "ts": 1},
                           {"type": "llm_call", "ts": 1}]) == {}


def test_build_timeline_sorted_ts_sequence():
    tl = build_timeline(_decision_events())
    steps = tl["a"]
    assert [s.ts for s in steps] == sorted(s.ts for s in steps)


def test_build_timeline_sorts_shuffled_events():
    events = _decision_events()[::-1]
    events.append({"type": "decision", "trace_id": "a", "phase": "route",
                   "data": {"strategy": "fallback"}})
    steps = build_timeline(events)["a"]
    assert [s.ts for s in steps] == [0, 10, 20, 30, 35, 40, 50]
    assert steps[0].detail["data"]["strategy"] == "fallback"


from agent.observability.report_data import build_timeline, group_stages


def _simple_events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q", "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 11,
         "data": {"in_domain": True, "intent": "q", "complexity": "low", "reason": "r"}},
        {"type": "decision", "trace_id": "a", "phase": "route", "ts": 12,
         "data": {"strategy": "teaching", "orchestrate": False}},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.teaching", "model": "m",
         "prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50,
         "latency_ms": 200, "status": "ok", "ts": 20},
        {"type": "trace_end", "trace_id": "a", "answer_len": 100, "total_llm_calls": 2,
         "total_tokens": 65, "total_latency_ms": 300.0, "reject": False, "ts": 30},
    ]


def _worker_events():
    return [
        {"type": "decision", "trace_id": "a", "phase": "orchestration.worker.1", "ts": 1,
         "data": {"task": "t1"}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
         "latency_ms": 10, "status": "ok", "ts": 2},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.worker.2", "ts": 3,
         "data": {"task": "t2"}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9,
         "latency_ms": 20, "status": "ok", "ts": 4},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13,
         "latency_ms": 30, "status": "ok", "ts": 5},
    ]


def test_group_stages_orders_stages_first_seen():
    stages = group_stages(build_timeline(_simple_events()))["a"]
    assert [s.title for s in stages] == ["classification", "route", "strategy.teaching", "result"]


def test_group_stages_classification_holds_decision_and_call():
    stages = group_stages(build_timeline(_simple_events()))["a"]
    classification = stages[0]
    assert [s.kind for s in classification.steps] == ["llm_call", "decision"]
    assert classification.steps[1].detail["data"]["intent"] == "q"


def test_group_stages_result_stage():
    stages = group_stages(build_timeline(_simple_events()))["a"]
    result = stages[-1]
    assert result.title == "result"
    assert result.steps[0].kind == "result"
    assert result.steps[0].detail["answer_len"] == 100


def test_group_stages_workers_collapse_into_one_stage():
    stages = group_stages(build_timeline(_worker_events()))["a"]
    assert len(stages) == 1
    assert stages[0].title == "orchestration.worker"
    assert [w.number for w in stages[0].workers] == [1, 2]
    assert stages[0].workers[0].task_title == "t1"
    assert stages[0].workers[1].task_title == "t2"
    assert len(stages[0].workers[0].steps) == 1
    assert len(stages[0].workers[1].steps) == 2


def test_group_stages_worker_title_set_from_decision_regardless_of_order():
    # llm_call arrives BEFORE the decision (current write-side ordering): the
    # worker title must still be captured from the decision event.
    events = [
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
         "latency_ms": 10, "status": "ok", "ts": 1},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.worker.1", "ts": 2,
         "data": {"task": "t1"}},
    ]
    stages = group_stages(build_timeline(events))["a"]
    assert stages[0].workers[0].task_title == "t1"


def test_group_stages_empty_timeline():
    assert group_stages({}) == {}


def test_group_stages_unknown_phase_own_stage():
    stages = group_stages(build_timeline([
        {"type": "llm_call", "trace_id": "a", "phase": "custom.phase", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 5, "status": "ok", "ts": 1},
    ]))["a"]
    assert [s.title for s in stages] == ["custom.phase"]


def test_build_timeline_critic_decision_type():
    tl = build_timeline([{"type": "decision", "trace_id": "a", "phase": "orchestration.critic.1",
                          "ts": 1, "data": {"task": "consistency"}}])
    steps = tl["a"]
    assert steps[0].kind == "decision"
    assert steps[0].detail["type"] == "worker"
    assert steps[0].detail["data"]["task"] == "consistency"


def test_group_stages_groups_critics_like_workers():
    events = [
        {"type": "trace_start", "trace_id": "a", "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.draft", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.critic.1", "ts": 20,
         "data": {"task": "consistency", "role": "R1"}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.critic.1", "model": "m",
         "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
         "latency_ms": 50, "status": "ok", "ts": 30},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.critic.2", "model": "m",
         "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
         "latency_ms": 50, "status": "ok", "ts": 40},
    ]
    stages = group_stages(build_timeline(events))["a"]
    assert [s.title for s in stages] == ["orchestration.draft", "orchestration.critic"]
    critic_stage = stages[1]
    assert [w.number for w in critic_stage.workers] == [1, 2]
    assert critic_stage.workers[0].task_title == "consistency"