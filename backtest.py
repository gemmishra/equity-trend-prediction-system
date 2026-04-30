import os
import pickle
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# -------------------- Suppress noisy warnings --------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*feature names.*")


# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- Model loading helpers --------------------
def load_model_pickle(name):
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_scaler(symbol):
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_scaler.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        return pickle.load(f)


# -------------------- yFinance column handling --------------------
def flatten_yfinance_columns(df):
    """
    Handles yfinance MultiIndex columns.

    Sometimes yfinance returns columns like:
    ('Close', 'RELIANCE.NS')

    This converts them back to:
    Close, Open, High, Low, Volume
    """

    if isinstance(df.columns, pd.MultiIndex):
        ohlcv_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        level_0 = set(df.columns.get_level_values(0))
        level_1 = set(df.columns.get_level_values(1))

        if len(level_0.intersection(ohlcv_cols)) > 0:
            df.columns = df.columns.get_level_values(0)
        elif len(level_1.intersection(ohlcv_cols)) > 0:
            df.columns = df.columns.get_level_values(1)

    return df


# -------------------- Feature engineering --------------------
def create_features(df):
    df = flatten_yfinance_columns(df)
    df = df.copy()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column from downloaded data: {col}")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Technical indicators / features
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["STD5"] = df["Close"].rolling(window=5).std()
    df["Volume_Change"] = df["Volume"].pct_change()

    # RSI calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Target: 1 if next day's close is higher than today's close, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = [
        "Return",
        "MA5",
        "MA10",
        "MA20",
        "STD5",
        "Volume_Change",
        "RSI",
    ]

    return df, features


# -------------------- Feature-name compatibility --------------------
def get_xgb_feature_order(xgb_model, fallback_features):
    """
    Gets exact feature names used by trained XGBoost model.
    Some saved models may contain trailing spaces in feature names.
    """

    try:
        model_features = xgb_model.get_booster().feature_names
    except Exception:
        model_features = None

    if model_features is None:
        model_features = fallback_features

    return list(model_features)


def get_rf_feature_order(rf_model, fallback_features):
    """
    Gets exact feature names used by trained Random Forest model, if available.
    """

    if hasattr(rf_model, "feature_names_in_"):
        return list(rf_model.feature_names_in_)

    return fallback_features


def get_scaler_feature_order(scaler, fallback_features):
    """
    Gets exact feature names used by fitted scaler, if available.
    """

    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    return fallback_features


def build_feature_frame(df, requested_features):
    """
    Builds a dataframe matching the exact feature names/order expected by a model.

    Fixes cases where saved models expect names like:
    'Return ', 'MA5 ', 'MA10 '

    but the current dataframe has:
    'Return', 'MA5', 'MA10'
    """

    output = pd.DataFrame(index=df.index)
    normalized_columns = {str(col).strip(): col for col in df.columns}

    for feature in requested_features:
        feature_clean = str(feature).strip()

        if feature in df.columns:
            source_col = feature
        elif feature_clean in normalized_columns:
            source_col = normalized_columns[feature_clean]
        else:
            raise ValueError(
                f"Missing feature '{feature}' in dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        # Preserve exact requested feature name for XGBoost compatibility
        output[feature] = df[source_col]

    return output


# -------------------- Backtesting logic --------------------
def backtest(symbol):
    print(f"\nBacktesting {symbol}...")

    # Load trained models
    rf = load_model_pickle(f"{symbol}_rf")
    xgb = load_model_pickle(f"{symbol}_xgb")
    lstm = load_model(os.path.join(MODEL_DIR, f"{symbol}_lstm.h5"))
    scaler = load_scaler(symbol)

    # Download historical stock data
    df = yf.download(
        f"{symbol}.NS",
        period="3y",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}. Check ticker symbol.")

    df, base_features = create_features(df)

    # Get model-specific feature orders
    xgb_features = get_xgb_feature_order(xgb, base_features)
    rf_features = get_rf_feature_order(rf, base_features)
    scaler_features = get_scaler_feature_order(scaler, base_features)

    window_size = 10

    if len(df) <= window_size + 1:
        raise ValueError(f"Not enough data for {symbol} after feature engineering.")

    test_indices = list(range(window_size, len(df) - 1))

    # -------------------- Prepare inputs --------------------
    # Random Forest was fitted without feature names, so pass NumPy array
    X_rf_all = build_feature_frame(df, rf_features).iloc[test_indices].to_numpy()

    # XGBoost is strict about feature names, so keep DataFrame
    X_xgb_all = build_feature_frame(df, xgb_features).iloc[test_indices]

    # MinMaxScaler was fitted without feature names, so pass NumPy array
    X_lstm_full = build_feature_frame(df, scaler_features).to_numpy()
    X_lstm_scaled = scaler.transform(X_lstm_full)

    X_lstm_sequences = np.array(
        [
            X_lstm_scaled[i - window_size + 1:i + 1]
            for i in test_indices
        ]
    )

    # -------------------- Batch predictions --------------------
    print("Generating model predictions...")

    rf_preds = np.ravel(rf.predict(X_rf_all)).astype(int)
    xgb_preds = np.ravel(xgb.predict(X_xgb_all)).astype(int)

    lstm_probs = np.ravel(lstm.predict(X_lstm_sequences, verbose=0))
    lstm_preds = (lstm_probs > 0.5).astype(int)

    final_signals = []

    for rf_pred, xgb_pred, lstm_pred in zip(rf_preds, xgb_preds, lstm_preds):
        votes = [int(rf_pred), int(xgb_pred), int(lstm_pred)]
        final_signal = 1 if votes.count(1) >= 2 else 0
        final_signals.append(final_signal)

    # -------------------- Simulated trading strategy --------------------
    initial_cash = 100000.0
    cash = initial_cash
    holdings = 0.0
    equity_curve = []

    closes = df["Close"].astype(float).values

    for signal, i in zip(final_signals, test_indices):
        price_today = float(closes[i])

        # Signal 1 = Buy / hold
        if signal == 1 and cash > 0:
            holdings = cash / price_today
            cash = 0.0

        # Signal 0 = Sell / stay in cash
        elif signal == 0 and holdings > 0:
            cash = holdings * price_today
            holdings = 0.0

        total_value = cash + holdings * price_today
        equity_curve.append(total_value)

    final_price = float(closes[-1])
    final_value = cash + holdings * final_price
    total_return = (final_value - initial_cash) / initial_cash * 100

    buy_signals = final_signals.count(1)
    sell_signals = final_signals.count(0)

    print(f"Completed backtest for {symbol}")
    print(f"Total signals: {len(final_signals)}")
    print(f"Buy/Hold signals: {buy_signals}")
    print(f"Sell/Cash signals: {sell_signals}")
    print(f"Final Value: Rs. {final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")

    # -------------------- Plot and save equity curve --------------------
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve)
    plt.title(f"{symbol} Strategy Equity Curve")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{symbol}_equity_curve.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved equity curve: {output_path}")

    plt.show()


# -------------------- Run backtest --------------------
if __name__ == "__main__":
    for stock_symbol in ["RELIANCE", "TCS", "INFY"]:
        backtest(stock_symbol)