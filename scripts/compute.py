import torch
from thop import profile, clever_format  # pip install thop
import time

from python.src.models.asymmetric_trinet import AsymmetricTriNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Asymmetric-TriNet ---
model = AsymmetricTriNet(num_classes=10).to(device).eval()

# Dummy inputs matching your dataset_builder output shapes
x_stft = torch.randn(1, 2, 513, 63).to(device)   # (B, 2, F, T) — adjust dims to match yours
x_iq   = torch.randn(1, 3, 1024).to(device)       # (B, 3, N)
x_if   = torch.randn(1, 1, 1024).to(device)       # (B, 1, N)

# FLOPs + Params
macs, params = profile(model, inputs=(x_stft, x_iq, x_if), verbose=False)
macs, params = clever_format([macs, params], "%.3f")
print(f"TriNet — Params: {params}, MACs: {macs}")

# Inference latency (GPU warm-up then timed)
N_RUNS = 500
with torch.no_grad():
    for _ in range(50): model(x_stft, x_iq, x_if)  # warm-up
    t0 = time.perf_counter()
    for _ in range(N_RUNS): model(x_stft, x_iq, x_if)
    t1 = time.perf_counter()
print(f"TriNet — Latency: {(t1-t0)/N_RUNS*1000:.2f} ms/sample")