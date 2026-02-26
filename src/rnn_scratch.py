# src/rnn_scratch.py
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    h_t = tanh(Wx x_t + Wh h_{t-1} + b)
    x: [B, T, d_in]
    returns:
      hs: [B, T, d_hidden]
      hT: [B, d_hidden]
    """
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden
        self.Wx = nn.Linear(d_in, d_hidden, bias=True)
        self.Wh = nn.Linear(d_hidden, d_hidden, bias=False)
        self.b  = nn.Parameter(torch.zeros(d_hidden))
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        assert x.dim() == 3, f"x must be [B,T,d_in], got {tuple(x.shape)}"
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        if h0 is None:
            h = torch.zeros(B, self.d_hidden, device=device, dtype=dtype)
        else:
            assert h0.shape == (B, self.d_hidden), f"h0 shape mismatch: {h0.shape}"
            h = h0

        hs = []
        for t in range(T):
            h = self.act(self.Wx(x[:, t, :]) + self.Wh(h) + self.b)
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # [B,T,H]
        return hs, h
