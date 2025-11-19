# S&P 500 Prediction Model

This project implements an advanced machine learning pipeline for predicting the daily direction of the S&P 500 index. It utilizes a hybrid approach combining deep learning models (Transformers, LSTMs) with sentiment analysis from news and social media, along with macroeconomic indicators.

## 🚀 Features

*   **Advanced Modeling**:
    *   **Transformer Architecture**: Uses a decoder-only Transformer with attention pooling for sequence encoding.
    *   **Hybrid Approach**: Preserves LSTM paths for temporal dynamics.
    *   **Ensemble Stacking**: A robust meta-model that ensembles sequence models and engineered features.
    *   **Tri-class Labeling**: Predicts Up, Down, or Flat (abstention) to filter out low-confidence days.

*   **Robust Data Ingestion**:
    *   Fetches market data via `yfinance` with retry logic (S&P 500, VIX, Treasury yields, Gold, Oil, etc.).
    *   Integrates macroeconomic data via **FRED API** (GDP, Unemployment, Inflation).
    *   incorporates sentiment analysis using **Reddit** (PRAW) and **Google Gemini** LLM.

*   **Feature Engineering**:
    *   Technical indicators: RSI, Moving Average Deltas, Volatility, Volume Ratios.
    *   Market regime detection based on VIX thresholds.

*   **Evaluation**:
    *   Walk-forward cross-validation to prevent look-ahead bias.
    *   Conformal prediction for calibrated confidence intervals.

## 📋 Prerequisites

*   Python 3.8+
*   API Keys (optional but recommended for full functionality):
    *   **FRED API Key**: For economic data.
    *   **Reddit Client ID & Secret**: For social sentiment.
    *   **Google Gemini API Key**: For advanced text analysis.

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    The project includes a `requirements.txt` file. You can install the necessary packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: The `setup_and_run.py` script will also attempt to automatically install missing dependencies.*

## ▶️ Usage

The easiest way to run the project is using the provided setup script, which handles environment configuration and execution.

### 1. Configure API Keys
Open `setup_and_run.py` and update the environment variables with your API keys:

```python
os.environ["FRED_API_KEY"] = "your_fred_api_key"
os.environ["REDDIT_CLIENT_SECRET"] = "your_reddit_secret"
os.environ["GEMINI_API_KEY"] = "your_gemini_key"
os.environ["REDDIT_CLIENT_ID"] = "your_reddit_client_id"
```

### 2. Run the Model
Execute the setup script:

```bash
python setup_and_run.py
```

This script will:
1.  Check and install required Python packages.
2.  Set up environment variables.
3.  Run the `improved_sp500_model.py` pipeline (training, evaluation, and prediction).

## 📂 Project Structure

*   `improved_sp500_model.py`: Main application logic containing data fetching, preprocessing, model architecture, and training loops.
*   `setup_and_run.py`: Wrapper script for easy setup and execution.
*   `requirements.txt`: List of Python dependencies.
*   `headlines.json`: Caches fetched news headlines for sentiment analysis.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Do not use this model as the sole basis for financial decisions. Stock market prediction is inherently uncertain, and past performance is not indicative of future results.
