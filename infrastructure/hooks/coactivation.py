"""Co-activation hook — Infrastructure adapter (legacy compatibility)."""

import torch


class CoActivationHook:
    """Captures co-activation statistics for one linear layer."""

    def __init__(self, weight_tensor, name):
        self.weight = weight_tensor.detach().clone()
        self.name = name
        self.freq_matrix = None
        self.total_tokens = 0
        out_dim, in_dim = weight_tensor.shape
        self.freq_matrix = torch.zeros(out_dim, in_dim, dtype=torch.float16)

    def hook(self, module, input, output):
        x = input[0]
        pre = output
        if x.dim() == 2:
            x = x.unsqueeze(0)
            pre = pre.unsqueeze(0)
        B, S, out_dim = pre.shape
        _, _, in_dim = x.shape
        active_out = (pre > 0).to(torch.float16)
        active_in = (x > 0).to(torch.float16)
        active_out_flat = active_out.reshape(-1, out_dim)
        active_in_flat = active_in.reshape(-1, in_dim)
        co = active_out_flat.T @ active_in_flat
        self.freq_matrix += co
        self.total_tokens += B * S

    def finalize(self):
        if self.total_tokens > 0:
            self.freq_matrix = self.freq_matrix.float()
            self.freq_matrix /= self.total_tokens
