# MOEA Portfolio Optimizer

Multi-Objective Portfolio Optimization using NSGA-II on NSE Nifty 50 stocks.

## How to Run

```bash
# Step 1 - Install dependencies
pip install -r requirements.txt

# Step 2 - Download historical data (one time only)
python data/fetch_stocks.py

# Step 3 - Run the dashboard
streamlit run dashboard/app.py
```

Open browser at: http://localhost:8501
