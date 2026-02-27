"""
Mock Trading Simulator - Enhanced Behavioral Economics A/B Testing
===================================================================
A Streamlit application for conducting double-blind A/B tests measuring
the impact of gamification on financial risk-taking behavior.

Enhanced Features:
- Response time tracking
- Portfolio value history
- Price direction detection
- Session duration tracking (2:30 limit)
- Additional psychometric measures
- Real-time analytics dashboard
- Anomaly detection

Author: Kavyan Jain
Date: 2026-02-06
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import time

# =============================================================================
# Configuration Constants
# =============================================================================
SESSION_TIME_LIMIT = 150  # 2 minutes 30 seconds

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Trading Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Session State Initialization
# =============================================================================
def initialize_session_state():
    """Initialize all session state variables if they don't exist."""
    
    # Core trading variables
    if 'user_id' not in st.session_state:
        st.session_state.user_id = random.randint(1000, 9999)
    
    if 'balance' not in st.session_state:
        st.session_state.balance = 10000.0
    
    if 'shares' not in st.session_state:
        st.session_state.shares = 0
    
    if 'price' not in st.session_state:
        st.session_state.price = 100.00
    
    if 'previous_price' not in st.session_state:
        st.session_state.previous_price = 100.00
    
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    if 'group' not in st.session_state:
        st.session_state.group = random.choice(["Control_Neutral", "Treatment_Gamified"])
    
    # Enhanced tracking variables
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = time.time()
    
    if 'trading_start_time' not in st.session_state:
        st.session_state.trading_start_time = None  # Set when trading begins
    
    if 'session_ended' not in st.session_state:
        st.session_state.session_ended = False
    
    if 'last_trade_time' not in st.session_state:
        st.session_state.last_trade_time = None
    
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = [{'time': datetime.now(), 'value': 10000.0}]
    
    if 'trade_sequence' not in st.session_state:
        st.session_state.trade_sequence = []  # List of 'B' or 'S'
    
    # Psychometric variables
    if 'psychometric_step' not in st.session_state:
        st.session_state.psychometric_step = 0
    
    if 'psychometric_completed' not in st.session_state:
        st.session_state.psychometric_completed = False
    
    if 'sensation_type' not in st.session_state:
        st.session_state.sensation_type = None
    
    if 'loss_aversion_score' not in st.session_state:
        st.session_state.loss_aversion_score = None
    
    if 'financial_literacy_score' not in st.session_state:
        st.session_state.financial_literacy_score = None
    
    if 'risk_tolerance' not in st.session_state:
        st.session_state.risk_tolerance = None
    
    if 'age_range' not in st.session_state:
        st.session_state.age_range = None
    
    if 'trading_experience' not in st.session_state:
        st.session_state.trading_experience = None

# =============================================================================
# Price Simulation (Random Walk)
# =============================================================================
def simulate_price():
    """
    Simulate stock price using a Random Walk model.
    Price fluctuates with each interaction using normal distribution.
    """
    st.session_state.previous_price = st.session_state.price
    change = np.random.normal(loc=0, scale=2)
    new_price = st.session_state.price + change
    st.session_state.price = max(1.0, round(new_price, 2))

def get_price_direction():
    """Determine if price is rising, falling, or stable."""
    diff = st.session_state.price - st.session_state.previous_price
    if diff > 0.5:
        return "RISING"
    elif diff < -0.5:
        return "FALLING"
    else:
        return "STABLE"

# =============================================================================
# Analytics & Metrics
# =============================================================================
def get_session_duration():
    """Get session duration in seconds."""
    return time.time() - st.session_state.session_start_time

def get_trading_duration():
    """Get trading session duration in seconds (after psychometric)."""
    if st.session_state.trading_start_time is None:
        return 0
    return time.time() - st.session_state.trading_start_time

def get_remaining_time():
    """Get remaining trading time in seconds."""
    if st.session_state.trading_start_time is None:
        return SESSION_TIME_LIMIT
    elapsed = get_trading_duration()
    remaining = SESSION_TIME_LIMIT - elapsed
    return max(0, remaining)

def is_session_expired():
    """Check if trading session has expired."""
    return get_remaining_time() <= 0

def get_response_time():
    """Get time since last trade in milliseconds."""
    if st.session_state.last_trade_time is None:
        return None
    return round((time.time() - st.session_state.last_trade_time) * 1000, 2)

def calculate_trade_patterns():
    """Analyze trade sequence patterns."""
    seq = st.session_state.trade_sequence
    if len(seq) < 2:
        return {"consecutive_buys": 0, "consecutive_sells": 0, "alternating_ratio": 0}
    
    consecutive_buys = max_consecutive(seq, 'B')
    consecutive_sells = max_consecutive(seq, 'S')
    alternating = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    alternating_ratio = alternating / (len(seq) - 1) if len(seq) > 1 else 0
    
    return {
        "consecutive_buys": consecutive_buys,
        "consecutive_sells": consecutive_sells,
        "alternating_ratio": round(alternating_ratio, 2)
    }

def max_consecutive(seq, value):
    """Find max consecutive occurrences of a value."""
    max_count = count = 0
    for item in seq:
        if item == value:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count

def detect_anomalies():
    """Detect anomalous trading behavior."""
    anomalies = []
    
    # Check for rapid trading (>5 trades in last minute)
    recent_trades = [t for t in st.session_state.trade_log 
                     if (datetime.now() - datetime.strptime(t['Timestamp'], "%Y-%m-%d %H:%M:%S")).seconds < 60]
    if len(recent_trades) > 5:
        anomalies.append("⚠️ Rapid Trading Detected (>5 trades/min)")
    
    # Check for excessive consecutive buys/sells
    patterns = calculate_trade_patterns()
    if patterns['consecutive_buys'] >= 5:
        anomalies.append("🔴 Excessive Consecutive Buys (FOMO behavior)")
    if patterns['consecutive_sells'] >= 5:
        anomalies.append("🔴 Excessive Consecutive Sells (Panic behavior)")
    
    return anomalies

# =============================================================================
# Trade Logging Function
# =============================================================================
def log_trade(action: str):
    """Log a trade event with all research data."""
    
    response_time = get_response_time()
    portfolio_value = st.session_state.balance + (st.session_state.shares * st.session_state.price)
    
    trade_record = {
        'User_ID': st.session_state.user_id,
        'Group': st.session_state.group,
        'Sensation_Type': st.session_state.sensation_type,
        'Loss_Aversion': st.session_state.loss_aversion_score,
        'Financial_Literacy': st.session_state.financial_literacy_score,
        'Risk_Tolerance': st.session_state.risk_tolerance,
        'Age_Range': st.session_state.age_range,
        'Experience': st.session_state.trading_experience,
        'Action': action,
        'Price': st.session_state.price,
        'Price_Direction': get_price_direction(),
        'Response_Time_Ms': response_time,
        'Portfolio_Value': round(portfolio_value, 2),
        'Session_Duration_Sec': round(get_session_duration(), 2),
        'Trade_Number': len(st.session_state.trade_log) + 1,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.trade_log.append(trade_record)
    st.session_state.last_trade_time = time.time()
    st.session_state.trade_sequence.append('B' if action == 'BUY' else 'S')
    st.session_state.portfolio_history.append({
        'time': datetime.now(),
        'value': portfolio_value
    })

# =============================================================================
# Trading Functions
# =============================================================================
def execute_buy():
    """Execute a buy order for 10 shares."""
    cost = st.session_state.price * 10
    
    if st.session_state.balance >= cost:
        st.session_state.balance -= cost
        st.session_state.shares += 10
        log_trade('BUY')
        return True, None
    else:
        return False, "Insufficient funds to complete this purchase."

def execute_sell():
    """Execute a sell order for 10 shares."""
    if st.session_state.shares >= 10:
        revenue = st.session_state.price * 10
        st.session_state.balance += revenue
        st.session_state.shares -= 10
        log_trade('SELL')
        return True, None
    else:
        return False, "Insufficient shares to complete this sale."

# =============================================================================
# Psychometric Assessment (Multi-Step)
# =============================================================================
def render_psychometric_screen():
    """Render the multi-step psychometric assessment."""
    
    st.markdown("---")
    st.header("📋 Pre-Trading Assessment")
    st.markdown("*Please complete this brief assessment before accessing the trading interface.*")
    
    # Progress bar
    progress = st.session_state.psychometric_step / 5
    st.progress(progress, text=f"Step {st.session_state.psychometric_step + 1} of 5")
    
    st.markdown("---")
    
    # Step 0: Sensation Seeking
    if st.session_state.psychometric_step == 0:
        st.subheader("1️⃣ Sensation Seeking")
        st.markdown("**Question:** *\"How do you feel about riding a big, fast roller coaster?\"*")
        
        response = st.radio(
            "Select your answer:",
            options=[
                "🎢 I love it! The scarier, the better.",
                "🚫 No thanks. I prefer to stay on the ground."
            ],
            index=None,
            key="sensation_response"
        )
        
        if st.button("Next →", type="primary"):
            if response is not None:
                st.session_state.sensation_type = "High" if "love it" in response else "Low"
                st.session_state.psychometric_step = 1
                st.rerun()
            else:
                st.error("Please select a response.")
    
    # Step 1: Loss Aversion
    elif st.session_state.psychometric_step == 1:
        st.subheader("2️⃣ Loss Aversion")
        st.markdown("**Statement:** *\"Losing $50 feels significantly worse than gaining $50 feels good.\"*")
        
        response = st.radio(
            "How much do you agree?",
            options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
            index=None,
            key="loss_aversion_response",
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.psychometric_step = 0
                st.rerun()
        with col2:
            if st.button("Next →", type="primary"):
                if response is not None:
                    score_map = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}
                    st.session_state.loss_aversion_score = score_map[response]
                    st.session_state.psychometric_step = 2
                    st.rerun()
                else:
                    st.error("Please select a response.")
    
    # Step 2: Financial Literacy
    elif st.session_state.psychometric_step == 2:
        st.subheader("3️⃣ Financial Literacy")
        st.markdown("**Question:** *Suppose you have $100 in a savings account earning 2% interest per year. After 5 years, how much would you have?*")
        
        response = st.radio(
            "Select your answer:",
            options=[
                "Exactly $110",
                "More than $110",
                "Less than $110",
                "I don't know"
            ],
            index=None,
            key="financial_literacy_response"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.psychometric_step = 1
                st.rerun()
        with col2:
            if st.button("Next →", type="primary"):
                if response is not None:
                    # Correct answer: "More than $110" (compound interest)
                    st.session_state.financial_literacy_score = 1 if response == "More than $110" else 0
                    st.session_state.psychometric_step = 3
                    st.rerun()
                else:
                    st.error("Please select a response.")
    
    # Step 3: Risk Tolerance
    elif st.session_state.psychometric_step == 3:
        st.subheader("4️⃣ Risk Tolerance")
        st.markdown("**On a scale of 1-10, how comfortable are you with financial risk?**")
        st.markdown("*1 = Very risk-averse (prefer guaranteed returns)*")
        st.markdown("*10 = Very risk-seeking (prefer high-risk/high-reward)*")
        
        response = st.slider(
            "Your risk tolerance:",
            min_value=1,
            max_value=10,
            value=5,
            key="risk_tolerance_slider"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.psychometric_step = 2
                st.rerun()
        with col2:
            if st.button("Next →", type="primary"):
                st.session_state.risk_tolerance = response
                st.session_state.psychometric_step = 4
                st.rerun()
    
    # Step 4: Demographics
    elif st.session_state.psychometric_step == 4:
        st.subheader("5️⃣ Demographics")
        
        age = st.selectbox(
            "Age Range:",
            options=["Select...", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            key="age_select"
        )
        
        experience = st.selectbox(
            "Trading Experience:",
            options=["Select...", "None", "Beginner (<1 year)", "Intermediate (1-5 years)", "Advanced (5+ years)"],
            key="experience_select"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.psychometric_step = 3
                st.rerun()
        with col2:
            if st.button("Start Trading 🚀", type="primary"):
                if age != "Select..." and experience != "Select...":
                    st.session_state.age_range = age
                    st.session_state.trading_experience = experience
                    st.session_state.psychometric_completed = True
                    st.session_state.trading_start_time = time.time()  # Start trading timer
                    st.session_state.last_trade_time = time.time()  # Start response tracking
                    st.rerun()
                else:
                    st.error("Please complete all fields.")

# =============================================================================
# Real-Time Analytics Dashboard (Sidebar)
# =============================================================================
def render_analytics_sidebar():
    """Render the real-time analytics dashboard in the sidebar."""
    
    with st.sidebar:
        st.header("📊 Live Analytics")
        
        # Countdown Timer (Most Important)
        remaining = get_remaining_time()
        remaining_mins = int(remaining // 60)
        remaining_secs = int(remaining % 60)
        
        if remaining > 30:
            st.success(f"⏱️ Time Remaining: **{remaining_mins}:{remaining_secs:02d}**")
        elif remaining > 10:
            st.warning(f"⏱️ Time Remaining: **{remaining_mins}:{remaining_secs:02d}**")
        else:
            st.error(f"⏱️ Time Remaining: **{remaining_mins}:{remaining_secs:02d}**")
        
        st.divider()
        
        # Session Info
        st.subheader("Session Info")
        st.metric("User ID", st.session_state.user_id)
        st.metric("Group", st.session_state.group.replace("_", " "))
        
        duration = get_trading_duration()
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.metric("Trading Duration", f"{minutes}m {seconds}s")
        
        st.divider()
        
        # Trade Statistics
        st.subheader("Trade Statistics")
        total_trades = len(st.session_state.trade_log)
        st.metric("Total Trades", total_trades)
        
        if total_trades > 0:
            buys = len([t for t in st.session_state.trade_log if t['Action'] == 'BUY'])
            sells = total_trades - buys
            st.metric("Buy/Sell Ratio", f"{buys}/{sells}")
            
            # Average response time
            response_times = [t['Response_Time_Ms'] for t in st.session_state.trade_log if t['Response_Time_Ms'] is not None]
            if response_times:
                avg_response = np.mean(response_times)
                st.metric("Avg Response Time", f"{avg_response:.0f}ms")
        
        st.divider()
        
        # Portfolio Performance
        st.subheader("Portfolio Performance")
        current_value = st.session_state.balance + (st.session_state.shares * st.session_state.price)
        pnl = current_value - 10000
        pnl_pct = (pnl / 10000) * 100
        
        st.metric("Portfolio Value", f"${current_value:,.2f}", delta=f"{pnl_pct:+.1f}%")
        
        # Portfolio chart
        if len(st.session_state.portfolio_history) > 1:
            df_portfolio = pd.DataFrame(st.session_state.portfolio_history)
            st.line_chart(df_portfolio.set_index('time')['value'], height=150)
        
        st.divider()
        
        # Trade Patterns
        st.subheader("Behavior Patterns")
        patterns = calculate_trade_patterns()
        st.caption(f"Max Consecutive Buys: {patterns['consecutive_buys']}")
        st.caption(f"Max Consecutive Sells: {patterns['consecutive_sells']}")
        st.caption(f"Alternating Ratio: {patterns['alternating_ratio']}")
        
        st.divider()
        
        # Anomaly Detection
        anomalies = detect_anomalies()
        if anomalies:
            st.subheader("⚠️ Anomalies Detected")
            for anomaly in anomalies:
                st.warning(anomaly)
        else:
            st.success("✅ No anomalies detected")

# =============================================================================
# Control Group Interface (Neutral)
# =============================================================================
def render_control_interface():
    """Render the neutral/control group trading interface."""
    
    simulate_price()
    
    st.title("Asset Allocation Terminal")
    st.markdown("---")
    
    # Portfolio Information
    st.subheader("Portfolio Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cash Balance", f"${st.session_state.balance:,.2f}")
    with col2:
        st.metric("Shares Held", f"{st.session_state.shares}")
    with col3:
        st.metric("Current Price", f"${st.session_state.price:.2f}")
    
    portfolio_value = st.session_state.balance + (st.session_state.shares * st.session_state.price)
    st.info(f"Total Portfolio Value: ${portfolio_value:,.2f}")
    
    st.markdown("---")
    
    # Trading Section
    st.subheader("Order Entry")
    st.caption("Each order trades 10 shares at the current market price.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Buy 10 Shares", type="secondary", use_container_width=True):
            success, error = execute_buy()
            if success:
                st.success("Order Executed")
            else:
                st.error(error)
    
    with col2:
        if st.button("Sell 10 Shares", type="secondary", use_container_width=True):
            success, error = execute_sell()
            if success:
                st.success("Order Executed")
            else:
                st.error(error)

# =============================================================================
# Treatment Group Interface (Gamified)
# =============================================================================
def render_gamified_interface():
    """Render the gamified/treatment group trading interface."""
    
    simulate_price()
    
    st.markdown("""
    <style>
    .gamified-header {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center;">🚀 MOONSHOT TRADING 🚀</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 20px; color: #ffd93d;">💰 Beat the Market! To the Moon! 💎🙌</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Portfolio Information with exciting styling
    st.markdown("### 🎯 Your Trading Power")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💵 War Chest", f"${st.session_state.balance:,.2f}")
    with col2:
        st.metric("📊 Diamond Hands", f"{st.session_state.shares} shares")
    with col3:
        direction = get_price_direction()
        emoji = "📈" if direction == "RISING" else "📉" if direction == "FALLING" else "➡️"
        st.metric(f"🔥 Hot Stock {emoji}", f"${st.session_state.price:.2f}")
    
    portfolio_value = st.session_state.balance + (st.session_state.shares * st.session_state.price)
    profit_loss = portfolio_value - 10000
    
    if profit_loss >= 0:
        st.success(f"🏆 TOTAL EMPIRE VALUE: ${portfolio_value:,.2f} | GAINS: +${profit_loss:,.2f} 📈")
    else:
        st.warning(f"💪 TOTAL EMPIRE VALUE: ${portfolio_value:,.2f} | Hold strong! ${profit_loss:,.2f}")
    
    st.markdown("---")
    
    # Trading Buttons
    st.markdown("### ⚡ EXECUTE YOUR MOVE")
    st.markdown("*Each legendary trade moves 10 shares!*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🟢 GO LONG")
        if st.button("🚀 BUY BUY BUY!", type="primary", use_container_width=True):
            success, error = execute_buy()
            if success:
                st.balloons()
                st.toast("🎉 POSITION ACQUIRED! You're going to the moon! 🌙", icon="🚀")
                st.success("✨ ORDER FILLED! Welcome aboard, astronaut! ✨")
            else:
                st.error(f"⚠️ {error}")
    
    with col2:
        st.markdown("##### 🔴 SECURE GAINS")
        if st.button("💰 SELL SELL SELL!", type="primary", use_container_width=True):
            success, error = execute_sell()
            if success:
                st.snow()
                st.toast("💸 PROFIT SECURED! 💸", icon="🎉")
                st.success("🏆 CHA-CHING! Gains locked in! 🏆")
            else:
                st.error(f"⚠️ {error}")
    
    st.markdown("---")
    
    # Motivational footer
    trades_count = len(st.session_state.trade_log)
    if trades_count >= 10:
        st.markdown("### 🔥 WHALE ALERT! You're on fire! 🔥")
    elif trades_count >= 5:
        st.markdown("### 💎 Diamond Hands in training! Keep going! 💎")
    elif trades_count >= 1:
        st.markdown("### 🌟 Great start! Your journey begins! 🌟")
    else:
        st.markdown("### 👆 Make your first trade above! The market awaits! 🎲")

# =============================================================================
# Data Export Section
# =============================================================================
def render_data_export():
    """Render the enhanced researcher data export section."""
    
    with st.expander("📊 Researcher Data"):
        st.markdown("### Enhanced Research Data Export")
        
        if len(st.session_state.trade_log) > 0:
            df = pd.DataFrame(st.session_state.trade_log)
            
            # Summary Statistics
            st.markdown("#### Session Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", len(df))
            with col2:
                buys = len(df[df['Action'] == 'BUY'])
                st.metric("Buy Orders", buys)
            with col3:
                sells = len(df[df['Action'] == 'SELL'])
                st.metric("Sell Orders", sells)
            with col4:
                avg_response = df['Response_Time_Ms'].dropna().mean()
                st.metric("Avg Response", f"{avg_response:.0f}ms" if pd.notna(avg_response) else "N/A")
            
            # Trade Patterns
            st.markdown("#### Behavioral Patterns")
            patterns = calculate_trade_patterns()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Consecutive Buys", patterns['consecutive_buys'])
            with col2:
                st.metric("Max Consecutive Sells", patterns['consecutive_sells'])
            with col3:
                st.metric("Alternating Ratio", f"{patterns['alternating_ratio']:.2f}")
            
            st.markdown("#### Complete Trade Log")
            st.dataframe(df, use_container_width=True)
            
            # CSV Download
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Enhanced Trade Log (CSV)",
                data=csv_data,
                file_name=f"enhanced_trade_log_user_{st.session_state.user_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Session Metadata
            st.markdown("#### Complete Session Metadata")
            st.json({
                "User_ID": st.session_state.user_id,
                "Experimental_Group": st.session_state.group,
                "Psychometrics": {
                    "Sensation_Type": st.session_state.sensation_type,
                    "Loss_Aversion_Score": st.session_state.loss_aversion_score,
                    "Financial_Literacy": st.session_state.financial_literacy_score,
                    "Risk_Tolerance": st.session_state.risk_tolerance
                },
                "Demographics": {
                    "Age_Range": st.session_state.age_range,
                    "Trading_Experience": st.session_state.trading_experience
                },
                "Session_Stats": {
                    "Final_Balance": st.session_state.balance,
                    "Final_Shares": st.session_state.shares,
                    "Final_Price": st.session_state.price,
                    "Total_Trades": len(st.session_state.trade_log),
                    "Session_Duration_Sec": round(get_session_duration(), 2)
                },
                "Behavioral_Patterns": patterns
            })
        else:
            st.info("No trades have been executed yet.")

# =============================================================================
# Session End Screen
# =============================================================================
def render_session_end():
    """Render the session end screen with final data export."""
    
    st.markdown("---")
    st.header("⏱️ Trading Session Complete!")
    st.markdown("*Your 2 minute 30 second trading session has ended. Thank you for participating!*")
    
    st.balloons()
    
    # Final Summary
    st.markdown("### 📊 Your Final Results")
    
    portfolio_value = st.session_state.balance + (st.session_state.shares * st.session_state.price)
    profit_loss = portfolio_value - 10000
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Balance", f"${st.session_state.balance:,.2f}")
    with col2:
        st.metric("Shares Held", st.session_state.shares)
    with col3:
        st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    with col4:
        delta_color = "normal" if profit_loss >= 0 else "inverse"
        st.metric("Profit/Loss", f"${profit_loss:,.2f}", delta=f"{(profit_loss/100):.1f}%")
    
    st.markdown("---")
    
    # Trade Summary
    total_trades = len(st.session_state.trade_log)
    if total_trades > 0:
        st.markdown("### 📈 Trading Activity")
        buys = len([t for t in st.session_state.trade_log if t['Action'] == 'BUY'])
        sells = total_trades - buys
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Buy Orders", buys)
        with col3:
            st.metric("Sell Orders", sells)
    
    st.markdown("---")
    
    # Data Export (always visible at end)
    st.markdown("### 📥 Download Your Data")
    
    if len(st.session_state.trade_log) > 0:
        df = pd.DataFrame(st.session_state.trade_log)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Complete Trade Log (CSV)",
            data=csv_data,
            file_name=f"final_trade_log_user_{st.session_state.user_id}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
        
        with st.expander("View Complete Trade Log"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No trades were executed during this session.")
    
    # Session Metadata
    with st.expander("📋 Complete Session Metadata (For Researchers)"):
        patterns = calculate_trade_patterns()
        st.json({
            "User_ID": st.session_state.user_id,
            "Experimental_Group": st.session_state.group,
            "Session_Duration_Sec": SESSION_TIME_LIMIT,
            "Psychometrics": {
                "Sensation_Type": st.session_state.sensation_type,
                "Loss_Aversion_Score": st.session_state.loss_aversion_score,
                "Financial_Literacy": st.session_state.financial_literacy_score,
                "Risk_Tolerance": st.session_state.risk_tolerance
            },
            "Demographics": {
                "Age_Range": st.session_state.age_range,
                "Trading_Experience": st.session_state.trading_experience
            },
            "Final_Stats": {
                "Final_Balance": st.session_state.balance,
                "Final_Shares": st.session_state.shares,
                "Final_Price": st.session_state.price,
                "Portfolio_Value": portfolio_value,
                "Profit_Loss": profit_loss,
                "Total_Trades": len(st.session_state.trade_log)
            },
            "Behavioral_Patterns": patterns
        })

# =============================================================================
# Main Application
# =============================================================================
def main():
    """Main application entry point."""
    
    initialize_session_state()
    
    if not st.session_state.psychometric_completed:
        render_psychometric_screen()
    elif is_session_expired() or st.session_state.session_ended:
        # Session has ended - show results
        st.session_state.session_ended = True
        render_session_end()
    else:
        # Active trading session
        # Auto-refresh every 1 second for timer
        st.empty()
        
        # Check if session just expired
        if is_session_expired():
            st.session_state.session_ended = True
            st.rerun()
        
        # Show analytics sidebar
        render_analytics_sidebar()
        
        # Render appropriate interface based on group
        if st.session_state.group == "Control_Neutral":
            render_control_interface()
        else:
            render_gamified_interface()
        
        # Data export section
        render_data_export()
        
        # Auto-refresh for countdown timer (every 1 second)
        time.sleep(1)
        st.rerun()

# =============================================================================
# Run Application
# =============================================================================
if __name__ == "__main__":
    main()
