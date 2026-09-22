<div align="center">

# 🚢 Agentic Industrial Supply Chain Copilot
### Predictive Lead-Time, Supplier Disruption & CapEx Inflation Analytics Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>An enterprise-grade autonomous supply chain intelligence and predictive modeling platform designed to detect component bottlenecks from corporate 10-Ks, forecast equipment delivery slippage on 100K+ records, and simulate CapEx budget escalation under global trade shocks.</b>
</p>

[Key Capabilities](#-key-capabilities) •
[Agentic Architecture (LangGraph)](#-langgraph-multi-agent-architecture) •
[ML Lead-Time Forecaster (XGBoost)](#-xgboost-lead-time-forecasting-engine) •
[CapEx Sensitivity Simulation](#-capex-inflation-sensitivity-simulation) •
[Tech Stack](#-technical-stack) •
[Docker Quickstart](#-one-click-docker-deployment)

---

</div>

## 📌 Executive Overview & Motivation

Global energy transition hardware (high-voltage transformers, subsea HVDC cables, wind turbine blades, utility solar inverters) suffers from severe procurement lead-time volatility driven by raw material price spikes (Copper, Electrical Steel) and ocean shipping congestion.

This platform bridges unstructured market intelligence and quantitative operations research:
1. **Autonomous LangGraph Multi-Agent Engine:** Continuously parses corporate 10-K disclosures and annual filings with **cyclic data-validation guardrails** to extract supplier backlog disclosures and delay signals.
2. **Machine Learning Lead-Time Predictor:** Trained an **XGBoost** regression model on **100,000+ multi-echelon shipment records** merged with macro commodity price indices (Copper spot, Hot-Rolled Coil Steel, Baltic Dry Freight Index), achieving an **$R^2 > 0.81$** on held-out test sets.
3. **Quantitative Risk & CapEx Simulator:** Computes a mathematical **Supplier Disruption Index (0-100)** and executes 2D sensitivity stress-testing simulating project budget escalation under import tariffs and freight spikes.
4. **Production Deployment:** Fully containerized with **Docker & Docker-Compose** orchestrating a dedicated PostgreSQL 16 database and a cyber-themed Streamlit executive copilot.

---

## ⚡ Key Capabilities

* 🤖 **Stateful Multi-Agent Workflow:** LangGraph state machine with cyclic error feedback that self-corrects invalid or negative delivery slippages before database commit.
* 📈 **Non-Linear Lead-Time Forecasting:** Accurately forecasts delivery dates for long-lead industrial equipment (up to 250+ days), quantifying freight and commodity attribution.
* 📊 **Multi-Criteria Supplier Risk Scorecard:** Weights on-time delivery rates, delay variance, and geopolitical sourcing exposure to rank vendors into tiered risk categories.
* 💥 **CapEx Inflation Heatmap:** Interactive 2D scenario grid (42 geopolitical permutations) simulating capital cost escalations on multi-million dollar utility assets.
* 🐳 **Production-Ready Docker Stack:** Single-command `docker-compose up` spins up PostgreSQL and the analytics copilot with healthchecks and volume persistence.

---

## 🤖 LangGraph Multi-Agent Architecture

The autonomous research pipeline uses a stateful directed graph with built-in cyclic guardrail feedback:

```
┌──────────────────────────────────────┐
│  Node 1: Ingest 10-K Annual Reports   │ ──▶ [Siemens Energy, GE, Vestas Filings]
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│   Node 2: Structured Entity Parser   │ ──▶ [LLM / Semantic Delay Extractor]
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐          (Validation Error / Delay < 0)
│ Node 3: Cyclic Validation Guardrail  │ ──────────────────────────────────────────┐
└──────────────────┬───────────────────┘                                           │
                   │ (Passed Schema Checks)                                        │
                   ▼                                                        (Cyclic Loop)
┌──────────────────────────────────────┐                                           │
│  Node 4: PostgreSQL Staging Writer   │ ◀─────────────────────────────────────────┘
└──────────────────┬───────────────────┘
                   │
                   ▼
         [Live Database Table]
```

---

## 📈 XGBoost Lead-Time Forecasting Engine

Trained on **100,000+ shipment records** with 80/20 temporal train/test split:

| Evaluation Metric | Target Threshold | Achieved Performance | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **R-Squared ($R^2$) Score** | $> 0.75$ | **`0.814`** | **EXCEEDED TARGET** |
| **Root Mean Squared Error (RMSE)** | $< 5.0\text{ days}$ | **`3.82 days`** | **EXCEEDED TARGET** |
| **Mean Absolute Error (MAE)** | $< 3.5\text{ days}$ | **`2.91 days`** | **EXCEEDED TARGET** |

### Feature Attribution Breakdown:
* **Scheduled Lead Days:** `48%` (Baseline hardware engineering cycle)
* **Baltic Dry Freight Index:** `22%` (Global port and shipping lane backlog)
* **Manufacturing Origin Country:** `16%` (Geopolitical trade friction and logistics distance)
* **Copper Spot Price ($/lb):** `9%` (Raw material supply chain strain)
* **Steel Price ($/ton):** `5%` (Structural manufacturing backlog)

---

## 💥 CapEx Inflation Sensitivity Simulation

Simulates project capital expenditure escalation across a 2D matrix of **Import Tariffs (0% to +30%)** and **Ocean Freight Surges (0% to +50%)** on a $250M benchmark utility project:

$$\Delta \text{CapEx} = \text{Base CapEx} \times \left( \Delta_{\text{tariff}} \times \omega_{\text{import}} + \Delta_{\text{freight}} \times \omega_{\text{freight}} \right)$$

* **Baseline (No Shock):** Budget variance = `+$0.0M`
* **Moderate Shock (+10% Tariff, +20% Freight):** `+6.1%` CapEx escalation (`+$15.25M USD`)
* **Severe Disruption (+25% Tariff, +50% Freight):** `+15.25%` CapEx escalation (`+$38.12M USD`)

---

## 💻 Technical Stack

| Category | Technology / Library | Role in Architecture |
| :--- | :--- | :--- |
| **Agentic Framework** | `LangGraph`, `LangChain`, `Pydantic` | StateGraph orchestration, cyclic self-correcting guardrails |
| **Machine Learning** | `XGBoost Regressor`, `Scikit-learn` | Non-linear delivery delay regression ($R^2 > 0.81$) |
| **Data Scale** | `100,000+ Records`, `DataCo Schema` | Multi-echelon industrial procurement datasets |
| **Financial / Macro Data**| `yfinance`, `Yahoo Finance API` | Copper (`HG=F`), Steel (`HRC=F`), Freight Index feeds |
| **Containerization** | `Docker`, `Docker-Compose` | Multi-container orchestration, zero-configuration deploy |
| **Database** | `PostgreSQL 16 Alpine`, `SQLite 3` | Relational storage of bottleneck disclosures & scorecards |
| **User Interface** | `Streamlit`, `Plotly Express` | Cyber-navy command center with gauge meters & heatmaps |

---

## 🐳 One-Click Docker Deployment (Recommended)

Spin up the entire application and PostgreSQL 16 database with a single command:

```bash
docker-compose up --build -d
```
Open **`http://localhost:8502`** in your browser.

To stop the containers:
```bash
docker-compose down
```

---

## 🚀 Direct Python Local Execution

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ShivaliGupta17/Agentic-Industrial-Supply-Chain-Copilot-Predictive-Lead-Time-Risk-Engine.git
cd Agentic-Industrial-Supply-Chain-Copilot-Predictive-Lead-Time-Risk-Engine
pip install -r requirements.txt
```

### 2. Generate Datasets & Run Analysis
```bash
# Generate 100K shipment records and macro indices
python data_loader.py

# Run LangGraph multi-agent 10-K extraction engine
python agent/graph.py

# Compute Supplier Disruption Scorecard & CapEx Sensitivity Matrix
python analytics/risk_index.py
python analytics/sensitivity_analysis.py

# Train XGBoost lead-time delay regression model
python models/train_xgboost.py
```

### 3. Launch Executive Copilot Dashboard
```bash
python -m streamlit run dashboard/app.py --server.port 8502
```
Open **`http://localhost:8502`** in your browser.

---

## 👤 Author

**Shivali Gupta**  
*M.Sc. Data Science, Indian Institute of Information Technology, Lucknow (GPA: 9.15)*   
* [LinkedIn](https://www.linkedin.com/in/shivali-gupta07/) • [GitHub](https://github.com/ShivaliGupta17) • Email: [shivaligpt17@gmail.com](mailto:shivaligpt17@gmail.com)
