import time
import torch
import numpy as np
import pandas as pd
import platform
import pennylane as qml

device = torch.device('cpu')
N_QUBITS = 4
q_dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def qcircuit_1layer(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for i in range(N_QUBITS):
        qml.Rot(weights[0, i, 0], weights[0, i, 1], weights[0, i, 2], wires=i)
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def qcircuit_2layers(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for l in range(2):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class AdaptiveHQNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveHQNN, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.LayerNorm(16),
            torch.nn.ReLU()
        )
        self.prelim_head = torch.nn.Linear(16, 1)
        self.to_latent = torch.nn.Sequential(
            torch.nn.Linear(16, N_QUBITS),
            torch.nn.Tanh()
        )
        self.qlayer1 = qml.qnn.TorchLayer(qcircuit_1layer, {"weights": (1, N_QUBITS, 3)})
        self.qlayer2 = qml.qnn.TorchLayer(qcircuit_2layers, {"weights": (2, N_QUBITS, 3)})
        self.q1_head = torch.nn.Linear(N_QUBITS, 1)
        self.q2_head = torch.nn.Linear(N_QUBITS, 1)

def benchmark_inference(input_dim=30, batch_size=1, n_warmup=10, n_repeats=100):
    model = AdaptiveHQNN(input_dim)
    model.eval()
    
    x = torch.randn(batch_size, input_dim)
    
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            h = model.encoder(x)
            p_logits = model.prelim_head(h)
            lat = model.to_latent(h) * np.pi
            _ = model.qlayer1(lat)
            _ = model.qlayer2(lat)
            
    # Benchmark Component 1: Classical Feature Encoder & Preliminary Head (Path 0)
    times_p0 = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            h = model.encoder(x)
            p_logits = model.prelim_head(h)
            p_prob = torch.sigmoid(p_logits)
            conf = torch.maximum(p_prob, 1.0 - p_prob)
        t1 = time.perf_counter()
        times_p0.append((t1 - t0) * 1000) # ms
        
    # Benchmark Component 2: Path 1 (Encoder + Prelim Head + Routing + 1 Q-Layer + Readout)
    times_p1 = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            h = model.encoder(x)
            p_logits = model.prelim_head(h)
            p_prob = torch.sigmoid(p_logits)
            conf = torch.maximum(p_prob, 1.0 - p_prob)
            lat = model.to_latent(h) * np.pi
            q1_out = model.qlayer1(lat).float()
            final_l = model.q1_head(q1_out)
        t1 = time.perf_counter()
        times_p1.append((t1 - t0) * 1000)
        
    # Benchmark Component 3: Path 2 (Encoder + Prelim Head + Routing + 2 Q-Layers + Readout)
    times_p2 = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            h = model.encoder(x)
            p_logits = model.prelim_head(h)
            p_prob = torch.sigmoid(p_logits)
            conf = torch.maximum(p_prob, 1.0 - p_prob)
            lat = model.to_latent(h) * np.pi
            q2_out = model.qlayer2(lat).float()
            final_l = model.q2_head(q2_out)
        t1 = time.perf_counter()
        times_p2.append((t1 - t0) * 1000)
        
    return {
        "p0_mean": np.mean(times_p0), "p0_std": np.std(times_p0),
        "p1_mean": np.mean(times_p1), "p1_std": np.std(times_p1),
        "p2_mean": np.mean(times_p2), "p2_std": np.std(times_p2),
    }

records = []
for bs in [1, 32]:
    for d, ds_name in [(30, 'WDBC Breast Cancer (D=30)'), (13, 'UCI Heart Disease (D=13)'), (8, 'Pima Diabetes (D=8)')]:
        res = benchmark_inference(input_dim=d, batch_size=bs, n_warmup=10, n_repeats=100)
        records.append({
            "Dataset": ds_name,
            "Batch Size": bs,
            "Path 0 Latency (ms)": f"{res['p0_mean']:.3f} ± {res['p0_std']:.3f}",
            "Path 1 Latency (ms)": f"{res['p1_mean']:.3f} ± {res['p1_std']:.3f}",
            "Path 2 Latency (ms)": f"{res['p2_mean']:.3f} ± {res['p2_std']:.3f}",
            "Path 0 vs Path 2 Speedup": f"{res['p2_mean'] / max(res['p0_mean'], 1e-6):.1f}x"
        })

df_bench = pd.DataFrame(records)
out_csv = "camera_ready/results/simulator_timing_benchmarks.csv"
df_bench.to_csv(out_csv, index=False)
print(f"\n[OK] Simulator timing benchmarks saved to {out_csv}")
print(df_bench.to_string(index=False))

# Environment details
print("\nBenchmarking Environment:")
print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
print(f"CPU: {platform.processor()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PennyLane Version: {qml.__version__}")
