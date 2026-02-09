<b>The app is deployed on Streamlit Cloud. Test it yourself:
https://vargamarton.streamlit.app/</b>

WORK IN PROGRESS

This is an interactive Streamlit Dashboard for portfolio optimization and risk analysis using Modern Portfolio Theory.

Key features:
- Portfolio optimization using SLSQP algo for max Sharpe and min volatility. Monte Carlo simulation
- Risk metrics: VaR, CVaR, Beta, Sharpe, Sortino, drawdown, skewness, kurtosis, risk budgeting 
- Stress testing with historical scenario analysis and correlation breakdown. Backtesting
- Geometric Brown Motion forecasting

Tech stack:
- Python 3.x
- Streamlit for UI
- NumPy+Pandas for data manipulation
- SciPy for SLSQP optimization
- Statsmodels for OLS regression
- yfinance for market data
- Plotly for visualization

Technical limitations: this app was made for educational purposes
- 252 trading days a year is assumed. Crypto and markets with other measurements will be off
- Daily close prices - no real time datastream
- Everything is calculated on historical data, no black swan events or regime changes are assumed
- Constant volatility is assumed
- Log normally distributed returns are assumed
- No slippage, no transaction costs, no bid-ask
- Exchange rate differencies between currencies aren't calculated

<img width="802" height="600" alt="newplot (2)" src="https://github.com/user-attachments/assets/0dadda86-9e2f-47ce-b557-005d08438d04" />
<hr>
<img width="344" height="224" alt="betareg" src="https://github.com/user-attachments/assets/bbe575f1-9398-431a-9ab3-b85130d5f007" />
<hr>
<img width="689" height="369" alt="pred" src="https://github.com/user-attachments/assets/6813ec64-2830-4b7d-81cf-f2aad5b32aa4" />
<hr>
<img width="713" height="383" alt="stresst" src="https://github.com/user-attachments/assets/67314798-baf4-47c3-b8d7-e23d8d585503" />
<hr>
<img width="648" height="354" alt="corr" src="https://github.com/user-attachments/assets/7b765e93-ca5d-4847-92a8-f13aea701241" />
<hr>
<img width="713" height="383" alt="rollingsharpe" src="https://github.com/user-attachments/assets/e14a3998-1c61-4ff4-ad60-804975d3deb1" />
<hr>
<img width="708" height="280" alt="riskcon" src="https://github.com/user-attachments/assets/9be3f358-ef74-4df5-8e7c-9fb2ec0a74d9" />
<hr>
<img width="701" height="348" alt="weight" src="https://github.com/user-attachments/assets/3d4cf858-1798-40e6-8271-1320ee1c3932" />
<hr>


Varga Márton
