"""HW1 extra credit: a GRU cell written gate by gate, verified numerically against nn.GRU.

    r = sigma(W_ir x + b_ir + W_hr h + b_hr)        reset gate   (0 = ignore old h)
    z = sigma(W_iz x + b_iz + W_hz h + b_hz)        update gate  (1 = keep old h)
    n = tanh (W_in x + b_in + r * (W_hn h + b_hn))  candidate state
    h' = (1 - z) * n + z * h

The gates are stored in PyTorch's layout, so a state_dict from nn.GRU can be loaded directly:
weight_ih / weight_hh have shape [3H, in] / [3H, H], with rows ordered (r, z, n)
(determined empirically: of the 6 orderings only this one matches nn.GRU).

Usage: python gru_scratch.py [run_dir_of_a_trained_gru ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


class GRUScratch(nn.Module):
    """One GRU layer, one explicit time step at a time."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.W_i = nn.Linear(in_dim, 3 * hidden)
        self.W_h = nn.Linear(hidden, 3 * hidden)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ir, iz, in_ = self.W_i(x).chunk(3, dim=-1)
        hr, hz, hn = self.W_h(h).chunk(3, dim=-1)
        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(in_ + r * hn)
        return (1 - z) * n + z * h

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, T, D]
        h = x.new_zeros(x.size(0), self.hidden)
        out = []
        for t in range(x.size(1)):
            h = self.step(x[:, t], h)
            out.append(h)
        return torch.stack(out, dim=1)


def copy_layer(src: nn.GRU, layer: int, dst: GRUScratch) -> None:
    with torch.no_grad():
        dst.W_i.weight.copy_(getattr(src, f"weight_ih_l{layer}"))
        dst.W_i.bias.copy_(getattr(src, f"bias_ih_l{layer}"))
        dst.W_h.weight.copy_(getattr(src, f"weight_hh_l{layer}"))
        dst.W_h.bias.copy_(getattr(src, f"bias_hh_l{layer}"))


def stack_forward(layers: list[GRUScratch], x: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        x = layer(x)
    return x


def compare(gru: nn.GRU, x: torch.Tensor) -> dict:
    """Max |difference| of outputs and of gradients wrt the input, scratch vs nn.GRU (float64, eval mode)."""
    gru = gru.double().eval()
    layers = [GRUScratch(gru.input_size if i == 0 else gru.hidden_size, gru.hidden_size).double()
              for i in range(gru.num_layers)]
    for i, layer in enumerate(layers):
        copy_layer(gru, i, layer)
    x = x.double()
    a_in = x.clone().requires_grad_(True)
    b_in = x.clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        a = gru(a_in)[0]
    b = stack_forward(layers, b_in)
    a.pow(2).sum().backward()
    b.pow(2).sum().backward()
    return {"output_max_abs_diff": (a - b).abs().max().item(),
            "grad_max_abs_diff": (a_in.grad - b_in.grad).abs().max().item(),
            "output_abs_scale": a.abs().mean().item(),
            "grad_abs_scale": a_in.grad.abs().mean().item()}


def main() -> None:
    res = {}
    torch.manual_seed(0)
    for L, D, H, T in [(1, 16, 32, 50), (2, 64, 128, 100), (3, 32, 32, 200)]:
        gru = nn.GRU(D, H, num_layers=L, batch_first=True)
        res[f"random_L{L}_D{D}_H{H}_T{T}"] = compare(gru, torch.randn(8, T, D))
    for run in sys.argv[1:]:
        ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=False)
        m = ck["config"]["model"]
        D = m["dim"]
        H = m["rnn_hidden"] or D
        gru = nn.GRU(D, H, num_layers=m["n_layers"], batch_first=True)
        gru.load_state_dict({k[4:]: v for k, v in ck["model"].items() if k.startswith("rnn.")})
        # real inputs: embeddings of random character ids
        E = ck["model"]["embed.weight"]
        ids = torch.randint(0, E.size(0), (16, 256), generator=torch.Generator().manual_seed(1))
        res[f"trained:{Path(run).name}"] = compare(gru, E[ids].float())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
