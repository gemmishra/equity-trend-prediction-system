# LSTM-Random-Forest-XGBoost-Stock-Predictor-with-Optuna
A hybrid AI-based stock market prediction system using LSTM, Random Forest, and XGBoost, built for real-world deployment with Optuna-powered tuning, feature-rich engineering, and ensemble prediction logic. Designed to optimize F1 score and accuracy, this system aims to generate reliable buy/sell signals on stocks. *still work under progress*
# 📈 LSTM + Random Forest + XGBoost Stock Predictor

---

## 🚀 About the Project
This project integrates:
- 🔁 **Recurrent Neural Networks (LSTM)** for sequential financial patterns
- 🌲 **Random Forest** for ensemble-based classification
- ⚡ **XGBoost** for gradient boosting decision trees
- 🎯 **Optuna** for automatic hyperparameter tuning (optional mode)
- 📊 **Backtesting Module** to simulate trading performance

> ⚙️ Built by a Computer Engineering student to demonstrate real-world ML/AI skills in finance and time series prediction.

---

## 📌 Features

- ✔️ Ensemble of 3 models: LSTM + RF + XGBoost
- ✔️ Flag-based retraining (no need to retrain every time)
- ✔️ Real stock data from Yahoo Finance
- ✔️ Feature-rich engineering: RSI, Moving Averages, Volatility, Volume
- ✔️ Backtesting for historical performance validation
- ✔️ Soft voting for final trade signal (BUY / SELL)
- ✔️ CLI-based output, no GUI bloat
- ✔️ Saved model reuse (`models/` folder)

---

## 🧠 Technologies Used

| Category         | Stack                                |
|------------------|---------------------------------------|
| Language         | Python 3.10                           |
| ML Models        | RandomForestClassifier, XGBClassifier |
| Deep Learning    | TensorFlow / Keras LSTM               |
| Optimization     | Optuna                                |
| Data Source      | yfinance                              |
| Indicators       | RSI, MA5/10/20, Volatility, Volume    |

---

## 📂 Project Structure
LSTM-RandomForest-XGBoost-Stock-Predictor/
├── LSTM+Random Forest+XGboost Stock Predictor.py
├── LSTM+Random Forest+XGboost Stock Predictor with optuna.py
├── backtest.py
├── models/ # Contains saved models
├── README.md
├── LICENSE # MIT License

---

## 🧪 How to Run

### ✅ 1. Install Dependencies
pip install -r requirements.txt
If no requirements.txt, install manually
pip install yfinance numpy pandas scikit-learn xgboost tensorflow optuna

---

### ✅ 2. Run Prediction Scripts
#### Without Optuna (default hyperparameters):
python "LSTM+Random Forest+XGboost Stock Predictor.py"

#### With Optuna (automatic tuning):
python "LSTM+Random Forest+XGboost Stock Predictor with optuna.py"

---

### ✅ 3. Backtest the System
python backtest.py
This will simulate trades and give statistics based on past 3 years of stock data.

---

## 📊 Output Example
📈 Processing RELIANCE...
✅ Trained RF | Accuracy: 0.72 | F1: 0.75  
✅ Trained XGB | Accuracy: 0.74 | F1: 0.77  
✅ Trained LSTM | Accuracy: 0.76 | F1: 0.79  
🔎 RF Signal: BUY  
🔎 XGB Signal: BUY  
🔎 LSTM Signal: SELL  
🟢 Final Signal: BUY  


---

## 🧠 Why This Project Matters

✅ Real-world applicability  
✅ Combines traditional ML and DL  
✅ Good F1/accuracy across multiple models  
✅ Shows model saving, loading, and production-like behavior  
✅ Demonstrates ability to tune, backtest, and validate system performance

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author
Arihanta Mishra  
B.Tech Computer Science and Engineering  
Siksha 'O' Anusandhan University

---

## 🙋‍♂️ Want to Hire Me?
If you're a company looking for a student with real-world skills in AI, ML, and applied engineering — this is just the start. Let's connect on [LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/aarav-mehta-16a183337?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)) or drop me a message!
