"""Score accumulators — Infrastructure adapters for all scoring methods.

Each function iterates over batches, collects activations/gradients,
and computes scores per layer.
"""

import time
import torch


def accumulate_gradient_scores(model, ffn_layers, dataset, num_batches, device):
    """|W| x |grad| — Forward + backward."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device="cpu")
        for name, module in ffn_layers
    }
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                score_accum[name] += (module.weight.data.abs() * module.weight.grad.abs()).cpu()
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return score_accum


def accumulate_wanda_scores(model, ffn_layers, dataset, num_batches, device):
    """|W| x ||X||_2 — Forward only."""
    act_accum = {
        name: torch.zeros(module.in_features, device="cpu")
        for name, module in ffn_layers
    }
    hooks, acts = [], {}

    def make_hook(n):
        def h(m, i, o):
            acts[n] = i[0].detach()
        return h

    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    t0, tokens_done = time.time(), 0
    with torch.no_grad():
        for bidx, batch in enumerate(dataset):
            if bidx >= num_batches:
                break
            ids = batch["input_ids"]
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            for name in act_accum:
                if name in acts:
                    act_accum[name] += acts[name].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
            tokens_done += ids.numel()
            if (bidx + 1) % 10 == 0:
                print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")

    for h in hooks:
        h.remove()
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")

    return {
        name: module.weight.data.float().cpu().abs()
        * (act_accum[name] / num_batches).unsqueeze(0)
        for name, module in ffn_layers
    }


def accumulate_magnitude_scores(model, ffn_layers, dataset, num_batches, device):
    """|W| — Naive baseline."""
    scores = {}
    for name, module in ffn_layers:
        s = module.weight.data.abs().cpu()
        s_max = s.max()
        scores[name] = s / s_max if s_max > 0 else s
    return scores


def accumulate_chain_scores(model, ffn_pairs, dataset, num_batches, device):
    """Chain scoring: gradient with downstream fc2 importance."""
    s1, s2 = {}, {}
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        done = set()
        for n1, fc1, n2, fc2 in ffn_pairs:
            if fc2.weight.grad is not None and n2 not in done:
                s2[n2] = s2.get(n2, 0) + (fc2.weight.data.abs() * fc2.weight.grad.abs()).cpu()
                done.add(n2)
            if fc1.weight.grad is not None and fc2.weight.grad is not None:
                imp = fc2.weight.grad.abs().sum(dim=0).cpu()
                imp = imp / (imp.max() + 1e-8)
                s1[n1] = s1.get(n1, 0) + (fc1.weight.data.abs() * fc1.weight.grad.abs()).cpu() * (1.0 + imp.unsqueeze(1))
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return {**s2, **s1}


def accumulate_wanda_chain_scores(model, ffn_pairs, dataset, num_batches, device):
    """Wanda Chain: activation-based with downstream importance."""
    act_fc1, act_fc2 = {}, {}
    hooks, activations = [], {}
    hooked = set()

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach()
        return hook_fn

    for name_fc1, fc1, name_fc2, fc2 in ffn_pairs:
        if fc1 not in hooked:
            hooks.append(fc1.register_forward_hook(make_hook(name_fc1)))
            hooked.add(fc1)
        if fc2 not in hooked:
            hooks.append(fc2.register_forward_hook(make_hook(name_fc2)))
            hooked.add(fc2)

    t0, tokens_done = time.time(), 0
    with torch.no_grad():
        for bidx, batch in enumerate(dataset):
            if bidx >= num_batches:
                break
            ids = batch["input_ids"]
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            batch_done = set()
            for name_fc1, fc1, name_fc2, fc2 in ffn_pairs:
                if name_fc2 in activations and name_fc2 not in batch_done:
                    act_fc2[name_fc2] = act_fc2.get(name_fc2, 0) + activations[name_fc2].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
                    batch_done.add(name_fc2)
                if name_fc1 in activations:
                    act_fc1[name_fc1] = act_fc1.get(name_fc1, 0) + activations[name_fc1].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
            tokens_done += ids.numel()
            if (bidx + 1) % 10 == 0:
                print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")

    for h in hooks:
        h.remove()
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")

    for d in (act_fc1, act_fc2):
        for k in d:
            d[k] = d[k] / num_batches

    all_scores, fc2_done = {}, set()
    for name_fc1, fc1, name_fc2, fc2 in ffn_pairs:
        if name_fc2 not in fc2_done and name_fc2 in act_fc2:
            all_scores[name_fc2] = fc2.weight.data.float().cpu().abs() * act_fc2[name_fc2].unsqueeze(0)
            fc2_done.add(name_fc2)
        if name_fc1 in act_fc1 and name_fc2 in act_fc2:
            imp = (fc2.weight.data.float().cpu().abs() * act_fc2[name_fc2].unsqueeze(0)).sum(dim=0)
            imp = imp / (imp.max() + 1e-8)
            all_scores[name_fc1] = fc1.weight.data.float().cpu().abs() * act_fc1[name_fc1].unsqueeze(0) * (1.0 + imp.unsqueeze(1))
    return all_scores


def accumulate_softmax_gradient_scores(model, ffn_layers, dataset, num_batches, device, temperature=1.0):
    """Softmax(|W| x |grad|) — Pure competition between connections."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device="cpu")
        for name, module in ffn_layers
    }
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                raw = (module.weight.data.abs() * module.weight.grad.abs()).cpu()
                probs = torch.softmax(raw.flatten() / temperature, dim=0).reshape_as(raw)
                score_accum[name] += probs
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return score_accum


def accumulate_weighted_softmax_scores(model, ffn_layers, dataset, num_batches, device, temperature=1.0, per_row=True):
    """|W| x softmax(|grad|) — Absolute weight x relativized gradient."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device="cpu")
        for name, module in ffn_layers
    }
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                w_abs = module.weight.data.abs().cpu()
                g_abs = module.weight.grad.abs().cpu()
                if per_row:
                    g_flat = g_abs.reshape(g_abs.shape[0], -1)
                    g_soft = torch.softmax(g_flat / temperature, dim=1).reshape_as(g_abs)
                else:
                    g_soft = torch.softmax(g_abs.flatten() / temperature, dim=0).reshape_as(g_abs)
                score_accum[name] += w_abs * g_soft
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return score_accum


# ======================================================================
#  APL-BASED SCORER — delegates to apl-pruning for ALL advanced methods
# ======================================================================

def accumulate_apl_scores(model, ffn_layers, dataset, num_batches, device,
                          formula_name, sample_tokens=500, needs_grad=False):
    """Generic accumulator: collects data, scores via APL.

    Works for: sparsegps, gcs_score, gps_cube, union_all3, latency_score,
    symmetry_score, fractal_score, and any other APL formula.
    """
    import numpy as np
    from infrastructure.scoring import get_apl_bridge

    APLScoringBridge = get_apl_bridge()
    bridge = APLScoringBridge(formula_name)

    layer_inputs = {name: [] for name, _ in ffn_layers}
    layer_outputs = {name: [] for name, _ in ffn_layers}
    grad_accum = {name: torch.zeros_like(module.weight.data, device="cpu") for name, module in ffn_layers}
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[f"{name}_in"] = input[0].detach()
            activations[f"{name}_out"] = output.detach()
        return hook_fn

    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    if needs_grad:
        model.train()
    else:
        model.eval()

    t0 = time.time()
    tokens_collected = 0

    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches or tokens_collected >= sample_tokens:
            break
        ids = batch["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        if needs_grad:
            model.zero_grad()
            loss = model(ids, labels=ids).loss
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(ids)

        for name, _ in ffn_layers:
            key_in = f"{name}_in"
            key_out = f"{name}_out"
            if key_in in activations:
                x_in = activations[key_in]
                if x_in.dim() == 3:
                    x_in = x_in.reshape(-1, x_in.shape[-1])
                layer_inputs[name].append(x_in.cpu())
            if key_out in activations:
                x_out = activations[key_out]
                if x_out.dim() == 3:
                    x_out = x_out.reshape(-1, x_out.shape[-1])
                layer_outputs[name].append(x_out.cpu())

        if needs_grad:
            for name, module in ffn_layers:
                if module.weight.grad is not None:
                    grad_accum[name] += module.weight.grad.detach().cpu()

        tokens_collected += ids.numel()
        if (bidx + 1) % 50 == 0:
            print(f"  Batch {bidx+1}/{num_batches} | {tokens_collected} tokens")

    for h in hooks:
        h.remove()
    print(f"  Collected {tokens_collected} tokens in {time.time()-t0:.0f}s")

    scores = {}
    for name, module in ffn_layers:
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        X_in = torch.cat(layer_inputs[name], dim=0)[:sample_tokens]
        W = module.weight.data.float()
        if X_in.shape[0] < 10:
            continue

        X_out = None
        if name in layer_outputs and len(layer_outputs[name]) > 0:
            X_out = torch.cat(layer_outputs[name], dim=0)[:sample_tokens]

        grad = None
        if needs_grad:
            grad = grad_accum[name] / num_batches

        try:
            score = bridge.calculate_from_raw(W=W, X_in=X_in, X_out=X_out, grad=grad, layer_name=name)
            if score is not None:
                s_max = score.max()
                scores[name] = (score / s_max).cpu() if s_max > 0 else score.cpu()
        except Exception as e:
            print(f"  {name}: APL scoring failed - {e}")

    return scores
