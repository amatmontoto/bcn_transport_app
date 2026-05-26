# Barcelona Transport Equity Dashboard
### ADAKA · CBI4AI · Challenge 2.2 with TMB

A **Graph Neural Network decision-support tool** for TMB network planners.  
It identifies which stops are underserving elderly residents, proposes feasible topology changes, and forecasts demographic demand through 2035 — so planners can see the consequences of a network modification *before* deploying it.

---

## What it does

The app has four tabs:

| Tab | What it shows |
|-----|--------------|
| **Network Map** | Every metro/FGC and bus stop, coloured by GNN elderly-coverage score |
| **Flagged Stops** | The stops under highest supply pressure — high elderly demand, low coverage |
| **Topology Proposals** | Feasible RELOCATE / ADD interventions ranked by coverage delta |
| **Predictive Layer** | Elderly-population forecast 2025 → 2035 per neighbourhood |

---

## How it works

```
Raw data (GTFS + demographics)
        ↓
  data_pipeline.py   →  builds the transport graph (2,629 nodes, 3,255 edges)
                         joins each stop to its neighbourhood
                         computes elderly catchment at 250m / 400m buffers
        ↓
  gnn_model.py       →  trains a GraphSAGE GNN to predict the coverage score
                         per stop; can re-score modified graphs (what-if engine)
        ↓
  proposals.py       →  flags stops under supply pressure
                         generates corridor-constrained RELOCATE / ADD proposals
        ↓
  forecast.py        →  fits a linear trend to 28 years of demographic data
                         projects elderly % per neighbourhood to 2030 / 2035
        ↓
  app.py             →  Streamlit dashboard — four interactive tabs
```

**Why a GNN and not just a formula?**  
Moving a stop changes the coverage of its neighbours too. The GNN propagates a modification across the graph through message passing, capturing network-wide ripple effects a per-stop formula cannot see. Validation: R² = 0.97, Pearson = 0.99.

**Why are the proposals feasible?**  
Every proposed coordinate lies on an existing GTFS route corridor — a path the current fleet already drives. No proposal can place a stop inside a building or imply new infrastructure.

---

## Project structure

```
bcn_transport_app/
│
├── app.py                  ← Streamlit dashboard (entry point)
├── data_pipeline.py        ← Graph construction + demographic join
├── gnn_model.py            ← GraphSAGE GNN definition + training
├── proposals.py            ← Flagging + topology proposals engine
├── forecast.py             ← Elderly population forecasting
├── requirements.txt        ← Python dependencies
│
├── data/                   ← Raw input files (see Data section below)
│   ├── data.zip
│   ├── TRANSPORTS.csv
│   ├── ESTACIONS_BUS.csv
│   ├── Population_per_age.csv
│   ├── Population_per_districte_i_barri.csv
│   └── 0301100100_UNITATS_ADM_POLIGONS.json
│
└── processed/              ← Auto-generated cache (created by the pipeline)
    ├── nodes.parquet
    ├── edges.parquet
    ├── shapes.geojson
    ├── barris.geojson
    ├── districts.geojson
    ├── demographics.parquet
    ├── forecast.parquet
    ├── gnn_model.pt
    ├── flagged.parquet
    └── proposals.parquet
```

> **Note:** The `processed/` folder is included in this repo so you can skip the pipeline step and run the app immediately. Re-run the pipeline only if you change the raw data.

---

## Prerequisites

- **Python 3.10 or higher** — download from [python.org](https://www.python.org/downloads/)  
  ⚠️ On Windows: when installing, check **"Add python.exe to PATH"** on the first screen.
- **~3 GB free disk space** (PyTorch is large)
- A terminal: PowerShell on Windows, Terminal on Mac/Linux

---

## Local setup

### 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/bcn_transport_app.git
cd bcn_transport_app
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

On Windows, if `pip` is not recognised:
```powershell
python -m pip install -r requirements.txt
```

> The first install takes a few minutes — PyTorch is ~2 GB.

### 3 — (Optional) Rebuild the processed data

Skip this step if the `processed/` folder already contains files — the app will work immediately.

Only run these if you want to rebuild from scratch or if you changed the raw data:

```bash
python data_pipeline.py   # ~2–3 min — builds the graph + catchments
python gnn_model.py       # ~1 min   — trains the GNN
python forecast.py        # instant  — fits the trend model
```

### 4 — Run the app

```bash
streamlit run app.py
```

On Windows:
```powershell
python -m streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## Data sources

All data is public. The `data/` folder must contain:

| File | Source | What it is |
|------|--------|------------|
| `data.zip` | [TMB Open Data](https://www.tmb.cat/en/about-tmb/open-data) | Full GTFS feed — stops, routes, trips, shapes, transfers |
| `TRANSPORTS.csv` | [Open Data BCN](https://opendata-ajuntament.barcelona.cat) | Metro and FGC station locations |
| `ESTACIONS_BUS.csv` | Open Data BCN | Bus stop locations |
| `Population_per_age.csv` | Open Data BCN / Idescat | Elderly % per neighbourhood, 1997–2025 |
| `Population_per_districte_i_barri.csv` | Open Data BCN | Population density per neighbourhood |
| `0301100100_UNITATS_ADM_POLIGONS.json` | Open Data BCN | Neighbourhood boundary polygons |

> ⚠️ The population CSVs must have **underscores** in their names (`Population_per_age.csv`, not `Population per age.csv`). If you re-download them, rename them accordingly.

---

## Deploying to AWS

The app is a standard Streamlit application — it runs anywhere Python runs. Two clean options:

---

### Option A — EC2 (recommended)

**1. Launch an EC2 instance**
- AWS Console → EC2 → Launch Instance
- OS: **Ubuntu 22.04 LTS**
- Instance type: `t3.small` or larger (PyTorch needs at least 2 GB RAM)
- Security Group: add an **Inbound rule** — Custom TCP, port **8501**, source `0.0.0.0/0`
- Download your `.pem` key file

**2. SSH into the instance**
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

**3. Install and run**
```bash
sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/YOUR_USERNAME/bcn_transport_app.git
cd bcn_transport_app
pip3 install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access at: **http://YOUR_EC2_PUBLIC_IP:8501**

**4. Keep it running after you close the terminal**
```bash
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

---

### Option B — Elastic Beanstalk (managed)

**1. Add a `Procfile`** to the repo root:
```
web: streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

**2. Deploy**
```bash
pip install awsebcli
eb init -p python-3.11 bcn-transport-app --region eu-west-1
eb create bcn-transport-env
eb open
```

> Use at least a `t3.small` instance under Configuration → Capacity.

---

### AWS cost estimate

| Resource | ~Monthly cost |
|----------|--------------|
| `t2.micro` (free tier, 12 months) | $0 |
| `t3.small` (after free tier) | ~$15 |
| 10 GB EBS storage | ~$1 |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip` not recognised on Windows | Use `python -m pip install -r requirements.txt` |
| `FileNotFoundError: processed/nodes.parquet` | Run the three pipeline scripts in order (Step 3) |
| `FileNotFoundError: data/0301100100...json` | Check all 6 files are in `data/` with exact names |
| Population CSVs not found | Rename files — must use underscores, not spaces |
| Port 8501 already in use | Run with `--server.port 8502` |
| Map tiles not loading | Check internet connection — tiles load from CartoDB at runtime |

---

## Model performance

| Metric | Value |
|--------|-------|
| Architecture | 2-layer GraphSAGE + regression head |
| Training epochs | 300 |
| Validation R² | 0.97 |
| Pearson correlation | 0.99 |
| Mean absolute error | 0.017 |

---

## Team

**ADAKA** · Group 03 · ESADE · CBI4AI 2026  
Amat · Aleksandra · Jakub · Diogenesis · Amaia  
Challenge 2.2 in collaboration with TMB — Transports Metropolitans de Barcelona
