"""Unit tests for lm_eval task registry."""

from domain.metrics.lm_eval_tasks import (
    TASK_REGISTRY,
    get_task_config,
    list_available_tasks,
    compute_mmlu_summary,
)


class TestTaskRegistry:
    def test_mmlu_exists(self):
        assert "mmlu" in TASK_REGISTRY
        assert len(TASK_REGISTRY["mmlu"]["tasks"]) == 57

    def test_all_tasks_have_config(self):
        for name in TASK_REGISTRY:
            cfg = TASK_REGISTRY[name]
            assert "tasks" in cfg
            assert "num_fewshot" in cfg

    def test_get_task_config(self):
        cfg = get_task_config("mmlu")
        assert cfg is not None
        assert cfg["num_fewshot"] == 5

    def test_get_unknown_task(self):
        assert get_task_config("unknown") is None

    def test_list_tasks(self):
        tasks = list_available_tasks()
        assert "mmlu" in tasks
        assert "lambada" in tasks
        assert len(tasks) >= 8


class TestComputeMMLUSummary:
    def test_basic(self):
        results = {"cat1": 0.8, "cat2": 0.6, "cat3": 0.7}
        summary = compute_mmlu_summary(results)
        assert summary["n_tasks"] == 3
        assert 0.6 < summary["avg"] < 0.8

    def test_single_category(self):
        results = {"cat1": 0.5}
        summary = compute_mmlu_summary(results)
        assert summary["avg"] == 0.5
        assert summary["std"] == 0.0

    def test_empty(self):
        summary = compute_mmlu_summary({})
        assert summary["n_tasks"] == 0
        assert summary["avg"] == 0.0
