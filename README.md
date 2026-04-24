# 🚀 NEON-BTC: Advanced Cryptocurrency Forecasting Portal

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NEON-BTC** is a sophisticated forecasting ecosystem designed to analyze and predict Bitcoin (BTC) price movements. Built with a modular architecture, it combines statistical rigor with machine learning to navigate the high volatility of the crypto market.

---

## 📂 1. Project Deliverables
This repository provides a complete end-to-end forecasting solution:
1.  **Streamlit Application (`app.py`)**: A production-ready script utilizing Streamlit Fragments for optimized performance.
2.  **Dataset Integration**: Built-in support for historical BTC time-series data.
3.  **Project Documentation**: Comprehensive guide on setup and model intuition.
4.  **Dependency Management**: A structured `requirements.txt` for consistent environment builds.

---

## 📊 2. Dataset Information
The portal is optimized for high-frequency financial data analysis.
* **Primary Source:** [Kaggle - Bitcoin Historical Data (2012–2026)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

---

## 🧠 3. Handling Crypto-Market Volatility
Bitcoin markets are defined by extreme heteroscedasticity and non-linear trends. **NEON-BTC** addresses these challenges using a multi-stage pipeline:

* **Log-Transformation:** Implementation of `np.log1p` to stabilize variance and normalize exponential price growth.
* **Iterative Differencing:** An interactive **Augmented Dickey-Fuller (ADF)** engine allows users to stabilize the mean by applying first and seasonal differencing ($d, D$).
* **Model Resilience:** * **Prophet:** Manages market outliers and "Black Swan" events using piecewise growth models.
    * **SARIMA:** Filters seasonal noise through moving average (MA) and auto-regressive (AR) components.
    * **Auto-ARIMA (StatsForecast):** Automatically optimizes hyper-parameters.

 ---

 ## 🛠️ 4. Tech Stack & Architecture
* **Engine:** Python 3.12+
* **Models:** `Statsmodels` (SARIMA), `Prophet` (Bayesian Inference), `StatsForecast` (Fast Auto-ARIMA).
* **UI/UX:** Streamlit with modular fragments for optimized state rendering.
* **Visualization:** Interactive Plotly charts with technical indicators like **SMA (20)** and **EMA (50)**.

---


## 📁 5. Repository Structure

```text
.
├── .streamlit/          # Streamlit configuration (UI/Server settings)
├── modules/             # Backend Intelligence & Logic
│   ├── processing.py    # Data cleansing, logging, and ADF Testing
│   ├── models.py        # ARIMA, Prophet, and Auto-ARIMA implementation
│   ├── evaluation.py    # Metrics calculation (MAE, RMSE)
│   ├── state.py         # Session State management logic
│   ├── theme.py         # Plotly custom visuals and styling
│   └── ui_components.py # Modular UI fragments and layouts
├── app.py               # Main Orchestration script (Streamlit entry point)
├── requirements.txt     # Python dependencies
├── .gitignore           # Git exclusion rules
└── LICENSE              # MIT License
```


---

## ⚙️ 7. Setup & Installation


### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### Create Environment

```bash
uv venv --python 3.12
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

##  8. Execution
To launch the portal and begin your analysis:

```bash
streamlit run app.py
```

---

## Authors

Course project implementation by:

 - Mohammed Sherif Safa







