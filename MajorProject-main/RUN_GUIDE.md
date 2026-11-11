# Adaptive HQNN Platform — Quick Start & Run Guide

This project includes a **React + TypeScript** web application and a **FastAPI** backend that runs real-time inference on the trained **Adaptive Hybrid Quantum-Classical Neural Network (Adaptive HQNN)** with PennyLane quantum circuit simulation.

---

## ⚡ Option 1: 1-Click Startup (Windows)

Simply double-click the **`start_platform.bat`** file in the root folder.

This script will automatically:
1. Check that Python and Node.js are installed.
2. Install any missing Python dependencies (`pip install -r backend/requirements.txt`).
3. Install frontend dependencies (`npm install` inside `frontend/`).
4. Start the FastAPI backend server on `http://127.0.0.1:8000`.
5. Start the React Vite development server on `http://localhost:5173`.
6. Open your default web browser directly to `http://localhost:5173`.

---

## 🛠️ Option 2: Manual Terminal Startup

If you prefer to start the servers manually in separate terminal windows:

### Step 1: Start the Backend (Terminal 1)
```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
* Backend API: `http://127.0.0.1:8000`
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### Step 2: Start the Frontend (Terminal 2)
```powershell
cd frontend
npm install
npm run dev
```
* Web Application: `http://localhost:5173`

---

## 🔬 What You Can Test in the Platform

1. **Live Diagnosis Page (`/diagnosis`)**:
   - Select between **WDBC Breast Cancer** (30 features), **UCI Cleveland Heart** (13 features), or **Pima Indians Diabetes** (8 features).
   - Click one of the instant **Clinical Presets**:
     - *Benign/Normal Sample* $\to$ Observe **Path 0 (0 Quantum Layers)** routing.
     - *Malignant/Severe Disease Sample* $\to$ Observe **Path 0 (0 Quantum Layers)** routing.
     - *Borderline Ambiguous Sample* $\to$ Observe dynamic escalation to **Path 1 (1 QL)** or **Path 2 (2 QL)** with live Pauli-$Z$ measurement expectations.
   - Click **"View SHAP Attributions"** to see real-time Kernel SHAP feature impact.
2. **Architecture Page (`/architecture`)**:
   - Inspect the mathematical equations, data flow, classical feature encoder, uncertainty router, and quantum variational layers.
3. **Quantum Circuit Page (`/circuit`)**:
   - Interactive gate-level circuit visualizer showing 4 qubit registers, $R_Y$ angle embeddings, single-qubit Euler rotations, and circular CNOT entanglement rings.
4. **Performance & Workload Dashboards (`/performance`, `/workload`)**:
   - Authoritative Table II and Table III benchmark metrics and quantum layer reduction charts (79.6% – 98.8% estimated quantum layer reduction).
