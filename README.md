# Behavioral Economics Mock Trading Simulator

A Streamlit-based A/B testing application for behavioral economics research, measuring the impact of gamification on financial risk-taking behavior.

## Features

- **Double-blind A/B Testing**: Random assignment to Control (Neutral) vs Treatment (Gamified) groups
- **Psychometric Assessment**: 5-step questionnaire capturing sensation seeking, loss aversion, financial literacy, risk tolerance, and demographics
- **Real-time Analytics**: Live dashboard with trade statistics, portfolio performance, and anomaly detection
- **Session Time Limit**: 2:30 minute controlled trading sessions
- **Data Export**: CSV download with comprehensive behavioral metrics

## Run Locally

```bash
pip install -r requirements.txt
streamlit run mock_trading_simulator.py
```

## Research Metrics Captured

- User ID, Experimental Group, Sensation Type
- Loss Aversion Score, Financial Literacy, Risk Tolerance
- Age Range, Trading Experience
- Trade Actions (BUY/SELL), Price, Price Direction
- Response Time (ms), Portfolio Value, Session Duration
- Trade Sequence Patterns, Anomaly Flags
