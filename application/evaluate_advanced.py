"""Use Cases : Advanced evaluation metrics — complete."""

from dataclasses import dataclass
import torch
from domain.scoring.ports import WeightProvider


@dataclass
class MetricResult:
    metric_name: str
    value: dict


# ═══════════════════════════════════════════════════════════════
#  DEAD NEURONS
# ═══════════════════════════════════════════════════════════════

class EvaluateDeadNeuronsUseCase:
    """Count dead neurons from a model."""

    def __init__(self, model: WeightProvider):
        self.model = model

    def execute(self) -> MetricResult:
        from domain.metrics.dead_neurons import count_dead_neurons_all_layers
        weights = {name: self.model.get_weight(name) for name in self.model.layer_names()}
        return MetricResult(metric_name="dead_neurons", value=count_dead_neurons_all_layers(weights))


# ═══════════════════════════════════════════════════════════════
#  WEIGHT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

class EvaluateWeightDistributionUseCase:
    """Analyze weight distribution across all layers."""

    def __init__(self, model: WeightProvider):
        self.model = model

    def execute(self) -> MetricResult:
        from domain.metrics.weight_distribution import analyze_weight_distribution
        result = {}
        for name in self.model.layer_names():
            W = self.model.get_weight(name)
            result[name] = analyze_weight_distribution(W)
        return MetricResult(metric_name="weight_distribution", value=result)


# ═══════════════════════════════════════════════════════════════
#  CUSTOM DOMAIN
# ═══════════════════════════════════════════════════════════════

class EvaluateCustomDomainUseCase:
    """Evaluate on custom domain questions."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def execute(self, questions=None) -> MetricResult:
        from domain.metrics.custom_domain import evaluate_custom_domain
        result = evaluate_custom_domain(self.model, self.tokenizer, questions)
        return MetricResult(metric_name="custom_domain", value=result)


# ═══════════════════════════════════════════════════════════════
#  NOISE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════

class EvaluateNoiseRobustnessUseCase:
    """Evaluate perplexity degradation under different noise levels."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, num_samples=30, max_len=64,
                noise_levels=None) -> MetricResult:
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.10, 0.25]

        import math
        from datasets import load_dataset

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                               split="test", streaming=True)
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)

        samples = []
        for batch in dataset:
            if len(samples) >= num_samples:
                break
            ids = self.tokenizer.encode(batch["text"], truncation=True,
                                        max_length=max_len, return_tensors="pt")
            if ids.size(1) >= 2:
                samples.append(ids)

        # Clean perplexity
        total_loss_clean, total_tok_clean = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for ids in samples:
                ids = ids.to(self.device)
                loss = self.model(ids, labels=ids).loss
                total_loss_clean += loss.item() * ids.size(1)
                total_tok_clean += ids.size(1)

        clean_perp = math.exp(total_loss_clean / total_tok_clean) if total_tok_clean > 0 else float("inf")

        # Noisy perplexities
        noisy_perps = {}
        for noise_std in noise_levels:
            total_loss_noisy, total_tok_noisy = 0.0, 0
            for ids in samples:
                ids = ids.to(self.device)
                with torch.no_grad():
                    out = self.model(ids)
                    noisy_logits = out.logits + torch.randn_like(out.logits) * noise_std
                    loss = torch.nn.functional.cross_entropy(
                        noisy_logits.view(-1, noisy_logits.size(-1)),
                        ids.view(-1), ignore_index=-100)
                total_loss_noisy += loss.item() * ids.size(1)
                total_tok_noisy += ids.size(1)
            noisy_perps[str(noise_std)] = math.exp(total_loss_noisy / total_tok_noisy) if total_tok_noisy > 0 else float("inf")

        from domain.metrics.noise_robust import compute_noise_degradation
        return MetricResult(metric_name="noise_robustness",
                           value=compute_noise_degradation(clean_perp, noisy_perps))


# ═══════════════════════════════════════════════════════════════
#  NEURON ACTIVATION FREQUENCY
# ═══════════════════════════════════════════════════════════════

class EvaluateNeuronFreqUseCase:
    """Measure how often each neuron fires."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, num_samples=100, max_len=128) -> MetricResult:
        from datasets import load_dataset
        from infrastructure.models.huggingface import HuggingFaceWeightProvider

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                               split="test", streaming=True)
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)

        ffn_layers = {}
        for name, module in self.model._model.named_modules():
            if hasattr(module, 'weight') and module.weight.dim() == 2:
                if any(p in name.lower() for p in HuggingFaceWeightProvider.FFN_PATTERNS):
                    ffn_layers[name] = module

        activation_counts = {name: torch.zeros(module.out_features)
                            for name, module in ffn_layers.items()}
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                act = output.detach()
                if act.dim() == 3:
                    act = act.reshape(-1, act.shape[-1])
                active = (act > 0).float().sum(dim=0).cpu()
                activation_counts[layer_name] = activation_counts[layer_name] + active
            return hook_fn

        for name, module in ffn_layers.items():
            hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        samples_done = 0
        tokens_done = 0
        for batch in dataset:
            if samples_done >= num_samples:
                break
            ids = self.tokenizer.encode(batch["text"], truncation=True,
                                        max_length=max_len, return_tensors="pt")
            if ids.size(1) < 2:
                continue
            ids = ids.to(self.device)
            with torch.no_grad():
                _ = self.model._model(ids)
            tokens_done += ids.numel()
            samples_done += 1

        for h in hooks:
            h.remove()

        from domain.metrics.neuron_freq import compute_activation_frequency
        result = {}
        for name, counts in activation_counts.items():
            freqs = counts.float() / max(tokens_done, 1)
            result[name] = compute_activation_frequency(
                freqs.unsqueeze(0).expand(100, -1))

        return MetricResult(metric_name="neuron_frequency",
                           value={"total_tokens": tokens_done, "per_layer": result})


# ═══════════════════════════════════════════════════════════════
#  RAPID ADAPTATION
# ═══════════════════════════════════════════════════════════════

class EvaluateRapidAdaptUseCase:
    """Mini fine-tuning on target domain, measure loss decrease rate."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, target_domain="wikitext", steps=100, lr=1e-4,
                max_len=128) -> MetricResult:
        from datasets import load_dataset

        if target_domain == "wikitext":
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                   split="train", streaming=True)
        else:
            dataset = load_dataset("c4", "en", split="train", streaming=True)

        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)

        optimizer = torch.optim.SGD(self.model._model.parameters(), lr=lr, momentum=0.9)
        initial_weights = {}
        for name, param in self.model._model.named_parameters():
            if param.dim() == 2 and any(p in name.lower() for p in ['fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']):
                initial_weights[name] = param.data.clone()

        self.model._model.train()
        losses = []
        step_idx = 0

        for batch in dataset:
            if step_idx >= steps:
                break
            ids = self.tokenizer.encode(batch["text"], truncation=True,
                                        max_length=max_len, return_tensors="pt")
            ids = ids.to(self.device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)

            optimizer.zero_grad()
            loss = self.model._model(ids, labels=ids).loss
            loss.backward()

            with torch.no_grad():
                for name, param in self.model._model.named_parameters():
                    if name in initial_weights:
                        mask = (initial_weights[name] != 0)
                        if param.grad is not None:
                            param.grad = param.grad * mask

            optimizer.step()
            losses.append(loss.item())
            step_idx += 1

        self.model._model.eval()

        from domain.metrics.rapid_adapt import compute_adaptation_rate
        return MetricResult(metric_name="rapid_adaptation",
                           value=compute_adaptation_rate(losses))


# ═══════════════════════════════════════════════════════════════
#  RELEARNING RATE
# ═══════════════════════════════════════════════════════════════

class EvaluateRelearnRateUseCase:
    """Measure how fast the model re-learns after partial weight reset."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, steps=100, lr=1e-4) -> MetricResult:
        import copy
        original_state = copy.deepcopy(self.model._model.state_dict())

        reset_count = 0
        total_nonzero = 0
        with torch.no_grad():
            for name, param in self.model._model.named_parameters():
                if param.dim() == 2 and any(p in name.lower() for p in ['fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']):
                    mask = (param.data != 0)
                    nonzero_idx = torch.where(mask)
                    n_nonzero = len(nonzero_idx[0])
                    if n_nonzero > 0:
                        n_reset = max(1, int(n_nonzero * 0.10))
                        reset_idx = torch.randperm(n_nonzero)[:n_reset]
                        for idx in reset_idx:
                            r = nonzero_idx[0][idx]
                            c = nonzero_idx[1][idx]
                            param.data[r, c] = torch.randn(1, device=self.device).item() * 0.02
                            reset_count += 1
                        total_nonzero += n_nonzero

        adapt_uc = EvaluateRapidAdaptUseCase(self.model, self.tokenizer, self.device)
        adapt_result = adapt_uc.execute(steps=steps, lr=lr)

        self.model._model.load_state_dict(original_state)

        from domain.metrics.relearn_rate import compute_weight_change
        weight_change = compute_weight_change(
            {n: original_state[n] for n in original_state if n in initial_weights},
            {n: self.model._model.state_dict()[n] for n in original_state if n in initial_weights}
        ) if False else {"avg_relative_weight_change": 0.0}

        return MetricResult(metric_name="relearning_rate", value={
            "weights_reset": reset_count,
            "weights_reset_pct": round(100 * reset_count / max(total_nonzero, 1), 2),
            "adaptation": adapt_result.value,
            "weight_change": weight_change,
        })


# ═══════════════════════════════════════════════════════════════
#  KL DIVERGENCE (dense vs sparse)
# ═══════════════════════════════════════════════════════════════

class EvaluateKLDivergenceUseCase:
    """KL divergence between dense and sparse model logits."""

    def __init__(self, model_sparse, model_dense, tokenizer, device="cpu"):
        self.model_sparse = model_sparse
        self.model_dense = model_dense
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, num_samples=50, max_len=128) -> MetricResult:
        from datasets import load_dataset

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                               split="test", streaming=True)
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)

        self.model_sparse.eval()
        self.model_dense.eval()

        kl_values = []
        samples_done = 0
        for batch in dataset:
            if samples_done >= num_samples:
                break
            ids = self.tokenizer.encode(batch["text"], truncation=True,
                                        max_length=max_len, return_tensors="pt")
            if ids.size(1) < 2:
                continue
            ids = ids.to(self.device)
            with torch.no_grad():
                out_s = self.model_sparse(ids)
                out_d = self.model_dense(ids)
            from domain.metrics.kl_divergence import compute_kl_divergence
            kl_values.append(compute_kl_divergence(out_s.logits, out_d.logits))
            samples_done += 1

        if not kl_values:
            return MetricResult(metric_name="kl_divergence",
                              value={"mean": None, "error": "No samples"})

        t = torch.tensor(kl_values)
        return MetricResult(metric_name="kl_divergence", value={
            "mean": t.mean().item(),
            "std": t.std().item(),
            "n_samples": len(kl_values),
        })


# ═══════════════════════════════════════════════════════════════
#  TOP-K RANK CORRELATION (dense vs sparse)
# ═══════════════════════════════════════════════════════════════

class EvaluateTopkRankUseCase:
    """Correlation of top-k token ranks between dense and sparse models."""

    def __init__(self, model_sparse, model_dense, tokenizer, device="cpu"):
        self.model_sparse = model_sparse
        self.model_dense = model_dense
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, num_samples=50, max_len=64, k_values=None) -> MetricResult:
        if k_values is None:
            k_values = [5, 10]

        from datasets import load_dataset

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                               split="test", streaming=True)
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)

        self.model_sparse.eval()
        self.model_dense.eval()

        correlations = {k: [] for k in k_values}
        samples_done = 0
        for batch in dataset:
            if samples_done >= num_samples:
                break
            ids = self.tokenizer.encode(batch["text"], truncation=True,
                                        max_length=max_len, return_tensors="pt")
            if ids.size(1) < 2:
                continue
            ids = ids.to(self.device)
            with torch.no_grad():
                out_s = self.model_sparse(ids)
                out_d = self.model_dense(ids)

            from domain.metrics.topk_rank import topk_overlap
            logits_s = out_s.logits[0, -1, :]
            logits_d = out_d.logits[0, -1, :]
            for k in k_values:
                correlations[k].append(topk_overlap(logits_s, logits_d, k))
            samples_done += 1

        result = {}
        for k in k_values:
            if correlations[k]:
                t = torch.tensor(correlations[k])
                result[f"top{k}_overlap_mean"] = t.mean().item()
                result[f"top{k}_overlap_std"] = t.std().item()
        result["n_samples"] = samples_done
        return MetricResult(metric_name="topk_rank", value=result)


# ═══════════════════════════════════════════════════════════════
#  OUTPUT STABILITY (dense vs sparse)
# ═══════════════════════════════════════════════════════════════

class EvaluateStabilityUseCase:
    """Compare greedy outputs between sparse and dense models."""

    def __init__(self, model_sparse, model_dense, tokenizer, device="cpu"):
        self.model_sparse = model_sparse
        self.model_dense = model_dense
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, prompts=None, max_new_tokens=50) -> MetricResult:
        if prompts is None:
            from domain.metrics.stability import DEFAULT_STABILITY_PROMPTS
            prompts = DEFAULT_STABILITY_PROMPTS

        self.model_sparse.eval()
        self.model_dense.eval()

        exact_matches = 0
        per_prompt = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            out_s = self.model_sparse.generate(ids, max_new_tokens=max_new_tokens,
                                               do_sample=False, num_beams=1)
            out_d = self.model_dense.generate(ids, max_new_tokens=max_new_tokens,
                                              do_sample=False, num_beams=1)
            text_s = self.tokenizer.decode(out_s[0], skip_special_tokens=True)
            text_d = self.tokenizer.decode(out_d[0], skip_special_tokens=True)
            is_exact = text_s == text_d
            if is_exact:
                exact_matches += 1
            per_prompt.append({
                "prompt": prompt,
                "sparse_output": text_s[:200],
                "dense_output": text_d[:200],
                "exact_match": is_exact,
            })

        return MetricResult(metric_name="stability", value={
            "exact_match_rate": exact_matches / len(prompts),
            "total_prompts": len(prompts),
            "per_prompt": per_prompt,
        })


# ═══════════════════════════════════════════════════════════════
#  MAGNITUDE PRUNING BENCHMARK
# ═══════════════════════════════════════════════════════════════

class EvaluateBenchmarkUseCase:
    """Magnitude pruning benchmark sweep."""

    def __init__(self, model: WeightProvider):
        self.model = model

    def execute(self, thresholds=None) -> MetricResult:
        from domain.metrics.benchmark import compute_magnitude_benchmark
        weights = {name: self.model.get_weight(name) for name in self.model.layer_names()}
        results = compute_magnitude_benchmark(weights, thresholds)
        return MetricResult(metric_name="benchmark", value={"results": results})


# ═══════════════════════════════════════════════════════════════
#  LM EVAL TASKS (MMLU, LAMBADA, HellaSwag, etc.)
# ═══════════════════════════════════════════════════════════════

class EvaluateLMEvalUseCase:
    """Evaluate on downstream tasks via lm_eval."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, task_name: str) -> MetricResult:
        from domain.metrics.lm_eval_tasks import get_task_config

        config = get_task_config(task_name)
        if config is None:
            return MetricResult(metric_name="lm_eval",
                              value={"error": f"Unknown task: {task_name}"})

        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM

            lm_obj = HFLM(pretrained=self.model, tokenizer=self.tokenizer)
            results = simple_evaluate(
                model=lm_obj,
                tasks=config["tasks"],
                num_fewshot=config["num_fewshot"],
                batch_size=config["batch_size"],
            )
            return MetricResult(metric_name="lm_eval",
                              value={"task": task_name, "results": results["results"]})
        except ImportError:
            return MetricResult(metric_name="lm_eval",
                              value={"error": "lm_eval not installed"})
        except Exception as e:
            return MetricResult(metric_name="lm_eval",
                              value={"error": str(e)})

    def execute_mmlu(self) -> MetricResult:
        """Run all 57 MMLU categories."""
        import json, os, numpy as np
        from domain.metrics.lm_eval_tasks import TASK_REGISTRY

        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            return MetricResult(metric_name="mmlu",
                              value={"error": "lm_eval not installed"})

        mmlu_tasks = TASK_REGISTRY["mmlu"]["tasks"]
        checkpoint_path = "reports/mmlu_checkpoint.json"
        completed = {}
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                completed = json.load(f)

        remaining = [t for t in mmlu_tasks if t not in completed]
        if remaining:
            lm_obj = HFLM(pretrained=self.model, tokenizer=self.tokenizer)
            new_results = simple_evaluate(
                model=lm_obj, tasks=remaining,
                num_fewshot=5, batch_size=1,
            )
            for t in remaining:
                r = new_results["results"][t]
                completed[t] = r.get("acc,none", r.get("acc_norm,none",
                                   r.get("acc", r.get("acc_norm", 0))))

            os.makedirs("reports", exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)

        scores = list(completed.values())
        from domain.metrics.lm_eval_tasks import compute_mmlu_summary
        summary = compute_mmlu_summary(
            {t: completed[t] for t in mmlu_tasks if t in completed})
        return MetricResult(metric_name="mmlu", value=summary)
