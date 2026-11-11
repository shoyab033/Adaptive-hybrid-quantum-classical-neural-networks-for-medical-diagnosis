import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

N_QUBITS = 4
MAX_QLAYERS = 2

# PennyLane Device (exact state-vector simulation)
q_dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def dynamic_quantum_circuit_1layer(inputs, weights):
    """
    Path 1: 1 Quantum Layer (4 Qubits)
    RY Angle Embedding -> 4 single-qubit Euler rotations (Rot) -> 4 circular CNOT entanglers -> Pauli-Z measurements.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for i in range(N_QUBITS):
        qml.Rot(weights[0, i, 0], weights[0, i, 1], weights[0, i, 2], wires=i)
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(q_dev, interface='torch', diff_method='backprop')
def dynamic_quantum_circuit_2layers(inputs, weights):
    """
    Path 2: 2 Quantum Layers (4 Qubits)
    RY Angle Embedding -> Layer 1 (Rot + CNOT) -> Layer 2 (Rot + CNOT) -> Pauli-Z measurements.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for l in range(MAX_QLAYERS):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class AdaptiveHQNN(nn.Module):
    """
    Adaptive Hybrid Quantum-Classical Neural Network with Dynamic Quantum Layer Selection.
    Architecture:
      - Classical Feature Encoder: Linear(D, 16) -> LayerNorm(16) -> ReLU()
      - Preliminary Classical Head: Linear(16, 1) -> Sigmoid -> Confidence C(x)
      - Latent Quantum Projection: Linear(16, 4) -> Tanh() -> scale by pi
      - Path 1 VQC: 1 Layer (4 Rot + 4 CNOT) -> Pauli-Z -> Linear(4, 1)
      - Path 2 VQC: 2 Layers (8 Rot + 8 CNOT) -> Pauli-Z -> Linear(4, 1)
    """
    def __init__(self, input_dim: int, n_qubits: int = N_QUBITS, max_layers: int = MAX_QLAYERS):
        super(AdaptiveHQNN, self).__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.max_layers = max_layers
        
        # 1. Classical Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        # 2. Preliminary Classical Head for confidence estimation
        self.prelim_head = nn.Linear(16, 1)
        
        # 3. Latent projection for Quantum Angle Embedding
        self.to_latent = nn.Sequential(
            nn.Linear(16, n_qubits),
            nn.Tanh()
        )
        
        # 4. Quantum Variational Layers
        self.qlayer1 = qml.qnn.TorchLayer(dynamic_quantum_circuit_1layer, {"weights": (1, n_qubits, 3)})
        self.qlayer2 = qml.qnn.TorchLayer(dynamic_quantum_circuit_2layers, {"weights": (max_layers, n_qubits, 3)})
        
        # 5. Quantum Readout Heads
        self.q1_head = nn.Linear(n_qubits, 1)
        self.q2_head = nn.Linear(n_qubits, 1)
        
    def forward_all_paths(self, x):
        """Forward pass computing all computational paths (used during joint multi-objective training)."""
        h = self.encoder(x)
        prelim_logits = self.prelim_head(h).squeeze(-1)
        latent = self.to_latent(h) * np.pi
        q1_out = self.qlayer1(latent).float()
        q2_out = self.qlayer2(latent).float()
        q1_logits = self.q1_head(q1_out).squeeze(-1)
        q2_logits = self.q2_head(q2_out).squeeze(-1)
        return prelim_logits, q1_logits, q2_logits
        
    def forward_adaptive(self, x, tau_high: float = 0.85, tau_low: float = 0.65):
        """
        Inference-time conditional forward pass:
        - Conf >= tau_high -> Path 0: Classical-only (0 Q-layers)
        - tau_low <= Conf < tau_high -> Path 1: 1 Quantum Layer
        - Conf < tau_low -> Path 2: 2 Quantum Layers
        """
        h = self.encoder(x)
        prelim_logits = self.prelim_head(h).squeeze(-1)
        prelim_probs = torch.sigmoid(prelim_logits)
        
        confidence = torch.maximum(prelim_probs, 1.0 - prelim_probs)
        latent = self.to_latent(h) * np.pi
        
        N = len(x)
        final_logits = torch.zeros(N, device=x.device)
        routes = np.zeros(N, dtype=int)
        quantum_outputs = [None] * N
        
        for i in range(N):
            conf_i = confidence[i].item()
            if conf_i >= tau_high:
                final_logits[i] = prelim_logits[i]
                routes[i] = 0
            elif conf_i >= tau_low:
                lat_i = latent[i:i+1]
                q1_out = self.qlayer1(lat_i).float()
                quantum_outputs[i] = q1_out.squeeze(0).cpu().detach().numpy().tolist()
                final_logits[i] = self.q1_head(q1_out).squeeze(-1)
                routes[i] = 1
            else:
                lat_i = latent[i:i+1]
                q2_out = self.qlayer2(lat_i).float()
                quantum_outputs[i] = q2_out.squeeze(0).cpu().detach().numpy().tolist()
                final_logits[i] = self.q2_head(q2_out).squeeze(-1)
                routes[i] = 2
                
        return final_logits, prelim_logits, confidence, routes, latent, quantum_outputs
