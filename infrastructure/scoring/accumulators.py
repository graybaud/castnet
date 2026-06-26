"""Score accumulators — Infrastructure adapters for all legacy scoring methods.

Each function iterates over batches, collects activations/gradients,
and delegates to domain strategies for per-layer calculation.
"""

import time
import torch
from domain.scoring.ports import WeightProvider, ActivationProvider


def accumulate_gradient_scores(
    model, ffn_layers, dataset, num_batches, device,
) -> dict:
    """|W| x |grad| — Forward + backward."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device='cpu')
        for name, module in ffn_layers
    }
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                score_accum[name] += (
                    module.weight.data.abs() * module.weight.grad.abs()
                ).cpu()
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return score_accum


def accumulate_wanda_scores(
    model, ffn_layers, dataset, num_batches, device,
) -> dict:
    """|W| x ||X||_2 — Forward only, state of the art."""
    act_accum = {
        name: torch.zeros(module.in_features, device='cpu')
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
            ids = batch['input_ids']
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            for name in act_accum:
                if name in acts:
                    act_accum[name] += (
                        acts[name].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
                    )
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


def accumulate_magnitude_scores(
    model, ffn_layers, dataset, num_batches, device,
) -> dict:
    """|W| — Naive baseline, no forward pass needed."""
    scores = {}
    for name, module in ffn_layers:
        s = module.weight.data.abs().cpu()
        s_max = s.max()
        scores[name] = s / s_max if s_max > 0 else s
    return scores


def accumulate_chain_scores(
    model, ffn_pairs, dataset, num_batches, device,
) -> dict:
    """Chain scoring: gradient with downstream fc2 importance."""
    s1, s2 = {}, {}
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
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
                s2[n2] = s2.get(n2, 0) + (
                    fc2.weight.data.abs() * fc2.weight.grad.abs()
                ).cpu()
                done.add(n2)
            if fc1.weight.grad is not None and fc2.weight.grad is not None:
                imp = fc2.weight.grad.abs().sum(dim=0).cpu()
                imp = imp / (imp.max() + 1e-8)
                s1[n1] = s1.get(n1, 0) + (
                    fc1.weight.data.abs() * fc1.weight.grad.abs()
                ).cpu() * (1.0 + imp.unsqueeze(1))
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"  Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    print(f"Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return {**s2, **s1}


def accumulate_wanda_chain_scores(
    model, ffn_pairs, dataset, num_batches, device,
) -> dict:
    """Wanda Chain: activation-based with downstream importance. Forward only."""
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
            ids = batch['input_ids']
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            batch_done = set()
            for name_fc1, fc1, name_fc2, fc2 in ffn_pairs:
                if name_fc2 in activations and name_fc2 not in batch_done:
                    act_fc2[name_fc2] = act_fc2.get(name_fc2, 0) + (
                        activations[name_fc2].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
                    )
                    batch_done.add(name_fc2)
                if name_fc1 in activations:
                    act_fc1[name_fc1] = act_fc1.get(name_fc1, 0) + (
                        activations[name_fc1].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
                    )
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
            all_scores[name_fc2] = (
                fc2.weight.data.float().cpu().abs()
                * act_fc2[name_fc2].unsqueeze(0)
            )
            fc2_done.add(name_fc2)
        if name_fc1 in act_fc1 and name_fc2 in act_fc2:
            imp = (
                fc2.weight.data.float().cpu().abs()
                * act_fc2[name_fc2].unsqueeze(0)
            ).sum(dim=0)
            imp = imp / (imp.max() + 1e-8)
            all_scores[name_fc1] = (
                fc1.weight.data.float().cpu().abs()
                * act_fc1[name_fc1].unsqueeze(0)
                * (1.0 + imp.unsqueeze(1))
            )
    return all_scores


def accumulate_softmax_gradient_scores(
    model, ffn_layers, dataset, num_batches, device, temperature=1.0,
) -> dict:
    """Softmax(|W| x |grad|) — Pure competition between connections."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device='cpu')
        for name, module in ffn_layers
    }
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
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


def accumulate_weighted_softmax_scores(
    model, ffn_layers, dataset, num_batches, device, temperature=1.0, per_row=True,
) -> dict:
    """|W| x softmax(|grad|) — Absolute weight x relativized gradient."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device='cpu')
        for name, module in ffn_layers
    }
    t0, tokens_done = time.time(), 0
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
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


# ═══════════════════════════════════════════════════════════
#  ADVANCED SCORERS — SparseGPS, SparseGPT, GCS, TRIDENT
# ═══════════════════════════════════════════════════════════

def accumulate_sparsegps_scores(
    model, ffn_layers, dataset, num_batches, device,
    sample_tokens=500, keep_fraction=None,
) -> dict:
    """SparseGPS: Energy x Unicity per connection. Forward-only."""
    import time
    from domain.scoring.geometric import find_natural_threshold

    layer_inputs = {name: [] for name, _ in ffn_layers}
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach()
        return hook_fn

    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    t0 = time.time()
    tokens_collected = 0

    with torch.no_grad():
        for bidx, batch in enumerate(dataset):
            if bidx >= num_batches or tokens_collected >= sample_tokens:
                break
            ids = batch['input_ids']
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            for name in activations:
                x = activations[name]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                layer_inputs[name].append(x.cpu())
            tokens_collected += ids.numel()

    for h in hooks:
        h.remove()

    masks = {}
    for name, module in ffn_layers:
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        X = torch.cat(layer_inputs[name], dim=0)[:sample_tokens].to(device)
        W = module.weight.data.float().to(device)
        if X.shape[0] < 10:
            continue

        # Simple energy-unicity scoring
        d_out, d_in = W.shape
        energy = (W.abs().unsqueeze(0) * X.unsqueeze(1)).abs().mean(dim=0)
        scores = energy / (energy.max() + 1e-8)
        threshold_info = find_natural_threshold({name: scores})
        threshold = threshold_info["threshold"]
        masks[name] = (scores >= threshold).float().cpu()

    return masks


def accumulate_sparsegpt_scores(
    model, ffn_layers, dataset, num_batches, device, sparsity=0.50,
) -> dict:
    """SparseGPT with Wanda+Gradient scoring. Forward+backward."""
    import time

    layer_inputs = {name: [] for name, _ in ffn_layers}
    grad_accum = {
        name: torch.zeros_like(module.weight.data, device='cpu')
        for name, module in ffn_layers
    }
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach()
        return hook_fn

    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.train()
    t0 = time.time()

    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        for name in activations:
            x = activations[name]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            layer_inputs[name].append(x.cpu())
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                grad_accum[name] += module.weight.grad.detach().cpu()

    for h in hooks:
        h.remove()

    masks = {}
    for name, module in ffn_layers:
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        X = torch.cat(layer_inputs[name], dim=0).float().to(device)
        W = module.weight.data.float()
        d_out, d_in = W.shape
        X_norm = X.norm(dim=0)
        grad_layer = grad_accum[name].abs().to(device) / num_batches
        n_keep = int(d_in * (1 - sparsity))
        mask = torch.zeros(d_out, d_in, device=device)

        for i in range(d_out):
            wanda_i = W[i].abs() * X_norm
            grad_i = W[i].abs() * grad_layer[i]
            w_max = wanda_i.max()
            if w_max > 0:
                wanda_i = wanda_i / w_max
            g_max = grad_i.max()
            if g_max > 0:
                grad_i = grad_i / g_max
            importance = torch.max(wanda_i, grad_i)
            _, top_indices = torch.topk(importance, n_keep)
            mask[i, top_indices] = 1.0

        masks[name] = mask.cpu()

    return masks


def accumulate_gcs_scores(
    model, ffn_layers, dataset, num_batches, device, sample_tokens=500,
) -> dict:
    """GCS: Geometric Complement Scoring — Wanda x (1 + irreplaceability)."""
    import time

    layer_inputs = {name: [] for name, _ in ffn_layers}
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach()
        return hook_fn

    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    t0 = time.time()
    tokens_collected = 0

    with torch.no_grad():
        for bidx, batch in enumerate(dataset):
            if bidx >= num_batches or tokens_collected >= sample_tokens:
                break
            ids = batch['input_ids']
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            for name in activations:
                x = activations[name]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                layer_inputs[name].append(x.cpu())
            tokens_collected += ids.numel()

    for h in hooks:
        h.remove()

    scores = {}
    for name, module in ffn_layers:
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        X = torch.cat(layer_inputs[name], dim=0)[:sample_tokens].to(device)
        W = module.weight.data.float().to(device)
        if X.shape[0] < 10:
            continue

        X_norm = X.norm(dim=0)
        wanda = W.abs() * X_norm.unsqueeze(0)
        # Simplified GCS: wanda with cosine uniqueness bonus
        X_n = X / (X.norm(dim=0, keepdim=True) + 1e-8)
        H_cos = (X_n.T @ X_n).abs()
        H_cos.fill_diagonal_(0.0)
        uniqueness = 1.0 - H_cos.mean(dim=1).unsqueeze(0)
        scores[name] = (wanda * (1.0 + uniqueness)).cpu()
        s_max = scores[name].max()
        if s_max > 0:
            scores[name] = scores[name] / s_max

    return scores


def accumulate_trident_scores(
    model, ffn_layers, dataset, num_batches, device, sample_tokens=500,
) -> dict:
    """TRIDENT: Union of 3 crossed methods. Forward+backward."""
    import time

    layer_inputs = {name: [] for name, _ in ffn_layers}
    grad_accum = {
        name: torch.zeros_like(module.weight.data, device='cpu')
        for name, module in ffn_layers
    }
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach()
        return hook_fn

    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.train()
    t0 = time.time()
    tokens_collected = 0

    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches or tokens_collected >= sample_tokens:
            break
        ids = batch['input_ids']
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        for name in activations:
            x = activations[name]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            layer_inputs[name].append(x.cpu())
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                grad_accum[name] += module.weight.grad.detach().cpu()
        tokens_collected += ids.numel()

    for h in hooks:
        h.remove()

    all_scores = {}
    for name, module in ffn_layers:
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        X = torch.cat(layer_inputs[name], dim=0)[:sample_tokens].to(device)
        W = module.weight.data.float().to(device)
        G = grad_accum[name].float().to(device) / num_batches
        X_norm = X.norm(dim=0)

        S_wanda = W.abs() * X_norm.unsqueeze(0)
        S_grad = W.abs() * G.abs()
        S_mag = W.abs()

        # TRIDENT union = max of the 3
        S_union = torch.max(torch.max(S_wanda / (S_wanda.max() + 1e-8),
                                       S_grad / (S_grad.max() + 1e-8)),
                            S_mag / (S_mag.max() + 1e-8))

        all_scores[name] = S_union.cpu()

    return all_scores
