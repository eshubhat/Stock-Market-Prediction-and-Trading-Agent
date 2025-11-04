"""
PortfolioCrafter: AI-Powered Autonomous Trading Research Platform
================================================================

This module provides the main Streamlit web interface for the PortfolioCrafter platform.
Users can generate AI-powered trading strategies, run performance simulations, and visualize results
through an intuitive dashboard.

Key Features:
- AI strategy generation using LLM agents
- Interactive performance simulation with real market data
- Comprehensive performance visualization
- Risk analysis and portfolio optimization
- Export capabilities for further analysis

Dependencies:
- Streamlit: Web application framework
- pandas/numpy: Data manipulation and numerical computing
- matplotlib: Visualization and charting
- Custom modules: Agent planning, performance simulation, data ingestion

Author: PortfolioCrafter Development Team
License: MIT
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional

# Internal module imports for core functionality
from src.agent.simple_planner import generate_strategy_proposals
from src.backtest.runner import run_backtest
from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime
from src.utils.config import config
from src.visualization.plots import (
    plot_portfolio_performance,
    plot_portfolio_composition,
    create_combined_plot,
    plot_strategy_formula,
    create_strategy_dashboard,
    get_timestamp_folder
)


# Configure Streamlit page settings for optimal user experience
st.set_page_config(
    page_title="PortfolioCrafter: AI Trading Research Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_available_assets() -> List[str]:
    """
    Load available assets from multiple sources for the dropdown.
    
    Combines assets from:
    1. Data store directory (cached parquet files)
    2. Config universe (predefined popular assets)
    3. Additional popular assets for comprehensive coverage
    
    Returns:
        List[str]: Comprehensive list of available asset symbols
        
    Note:
        Prioritizes cached assets but includes all config universe assets
        for maximum flexibility and user choice.
    """
    # Start with assets from config universe
    config_assets = config.get('universe', [])
    
    # Add assets from data store (cached files)
    data_dir = os.path.join(os.getcwd(), "data_store")
    cached_assets = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".parquet"):
                # Extract symbol by removing .parquet extension
                symbol = file.split(".")[0]
                cached_assets.append(symbol)
    
    # Combine and deduplicate
    all_assets = list(set(config_assets + cached_assets))
    
    # Add additional popular assets that might not be in config
    additional_assets = [
        # More ETFs
        "DIA", "VOO", "IVV", "VUG", "VTV", "VYM", "SCHD", "ARKK", "ARKQ",
        # More Tech
        "AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "IBM", "UBER", "LYFT",
        # More Financial
        "C", "AXP", "BLK", "SCHW", "USB", "PNC", "TFC", "COF",
        # More Healthcare
        "GILD", "AMGN", "BIIB", "REGN", "VRTX", "ILMN", "MRNA", "BNTX",
        # More Consumer
        "HD", "LOW", "TGT", "COST", "SBUX", "NKE", "DIS", "CMCSA",
        # More Energy
        "EOG", "PXD", "MPC", "VLO", "PSX", "KMI", "WMB",
        # More Industrial
        "HON", "UPS", "FDX", "LMT", "RTX", "NOC", "GD",
        # More International
        "EWG", "EWC", "EWA", "EWH", "EWS", "EWT", "EWY",
        # More Crypto
        "GBTC", "ETHE", "MSTR", "RIOT", "MARA",
        # Commodities
        "USO", "UNG", "SLV", "PPLT", "PALL",
        # Volatility
        "VXX", "UVXY", "SVXY", "TVIX"
    ]
    
    # Combine all assets and remove duplicates while preserving order
    final_assets = []
    seen = set()
    
    # Add cached assets first (these are already downloaded)
    for asset in cached_assets:
        if asset not in seen:
            final_assets.append(asset)
            seen.add(asset)
    
    # Add config universe assets
    for asset in config_assets:
        if asset not in seen:
            final_assets.append(asset)
            seen.add(asset)
    
    # Add additional popular assets
    for asset in additional_assets:
        if asset not in seen:
            final_assets.append(asset)
            seen.add(asset)
    
    return final_assets


def load_available_strategies() -> List[str]:
    """
    Load the list of available strategy types supported by the platform.
    
    This function returns the strategy types that are implemented in the
    multi_strategy module and can be executed by the performance simulation engine.
    
    Returns:
        List[str]: List of strategy names available for selection
        
    Strategy Types:
        - momentum: Moving average crossover and trend following
        - mean_reversion: Bollinger Bands and RSI-based signals  
        - volatility: Volatility targeting and VIX-based strategies
        - trend_following: Directional trend capture strategies
        - breakout: Range breakout and momentum strategies
        - regime_based: Market regime adaptive allocation
    """
    return [
        "momentum",
        "mean_reversion",
        "volatility",
        "trend_following",
        "breakout",
        "regime_based"
    ]


def optimize_strategy_parameters(strategy_info, data, num_trials=50):
    """
    Perform hyperparameter optimization for a strategy.
    
    Args:
        strategy_info: Dictionary with strategy information
        data: DataFrame with market data
        num_trials: Number of optimization trials
        
    Returns:
        Dictionary with optimized parameters
    """
    strategy_type = strategy_info["strategy_type"]
    assets = strategy_info["asset_tickers"]
    params = strategy_info["params"].copy()
    
    # Define parameter search spaces based on strategy type
    param_spaces = {}
    
    if strategy_type == "momentum":
        param_spaces = {
            "fast_window": {"min": 5, "max": 30},
            "slow_window": {"min": 30, "max": 100}
        }
    elif strategy_type == "mean_reversion":
        param_spaces = {
            "window": {"min": 10, "max": 60},
            "num_std": {"min": 1.0, "max": 3.0, "step": 0.2}
        }
    elif strategy_type == "volatility":
        param_spaces = {
            "window": {"min": 10, "max": 60},
            "vol_threshold": {"min": 0.01, "max": 0.05, "step": 0.005}
        }
    
    # Create trials with different parameter combinations
    best_sharpe = -np.inf
    best_params = params.copy()
    results = []
    
    for _ in range(num_trials):
        trial_params = params.copy()
        
        # Generate random parameters within the search space
        for param, space in param_spaces.items():
            if param in trial_params:
                if isinstance(trial_params[param], int):
                    trial_params[param] = np.random.randint(space["min"], space["max"])
                else:
                    step = space.get("step", 0.1)
                    trial_params[param] = np.random.choice(
                        np.arange(space["min"], space["max"] + step, step)
                    )
        
        # Run backtest with the trial parameters
        trial_info = strategy_info.copy()
        trial_info["params"] = trial_params
        
        try:
            backtest_result = run_backtest(
                data,
                trial_info["asset_tickers"],
                trial_info["strategy_type"],
                trial_info["params"],
                trial_info.get("allocation_weights")
            )
            
            sharpe = backtest_result.get("metrics", {}).get("Sharpe Ratio", -np.inf)
            
            # Store result
            result = {
                "params": trial_params,
                "sharpe": sharpe
            }
            results.append(result)
            
            # Update best parameters if better Sharpe ratio
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = trial_params.copy()
        
        except Exception as e:
            st.error(f"Error during optimization trial: {str(e)}")
            continue
    
    # Return the optimized parameters
    return {
        "optimized_params": best_params,
        "trials": results,
        "best_sharpe": best_sharpe
    }


def main():
    # Add sidebar
    st.sidebar.title("PortfolioCrafter Dashboard")
    
    # Load available assets and strategies
    available_assets = load_available_assets()
    available_strategies = load_available_strategies()
    
    # Sidebar inputs
    st.sidebar.header("📊 Analysis Settings")
    
    # Date range selection
    today = datetime.now()
    default_end_date = today - timedelta(days=1)
    default_start_date = default_end_date - timedelta(days=365*3)  # 3 years
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        max_value=default_end_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        max_value=today
    )
    
    # Asset selection with categories
    st.sidebar.subheader("📊 Asset Categories")
    
    # Create asset categories for better organization
    asset_categories = {
        "🏛️ Major ETFs": ["SPY", "QQQ", "IWM", "TLT", "GLD", "VTI", "VEA", "VWO", "BND", "VNQ", "DIA", "VOO", "IVV"],
        "💻 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC", "CRM", "ADBE", "ORCL"],
        "🏦 Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC"],
        "🏥 Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "GILD", "AMGN", "BIIB", "REGN", "VRTX", "ILMN"],
        "🛒 Consumer": ["WMT", "PG", "KO", "PEP", "MCD", "HD", "LOW", "TGT", "COST", "SBUX", "NKE", "DIS"],
        "⚡ Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "KMI"],
        "🏭 Industrial": ["BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "NOC"],
        "🌍 International": ["EWJ", "EWZ", "FXI", "EWU", "EWG", "EWC", "EWA", "EWH", "EWS", "EWT"],
        "₿ Crypto": ["BTC-USD", "ETH-USD", "COIN", "GBTC", "ETHE", "MSTR", "RIOT", "MARA"],
        "🥇 Commodities": ["GLD", "SLV", "USO", "UNG", "PPLT", "PALL"],
        "📈 Volatility": ["VXX", "UVXY", "SVXY", "TVIX"]
    }
    
    # Preset portfolio options
    preset_portfolios = {
        "🏆 S&P 500 Core": ["SPY", "QQQ", "IWM", "TLT"],
        "💻 Tech Heavy": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
        "🏦 Financial Focus": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP"],
        "🌍 Global Diversified": ["SPY", "VEA", "VWO", "EWJ", "EWZ", "FXI", "GLD"],
        "⚖️ Balanced Growth": ["SPY", "QQQ", "VTI", "BND", "VNQ", "GLD"],
        "🚀 High Growth": ["QQQ", "ARKK", "TSLA", "NVDA", "META", "NFLX"],
        "🛡️ Conservative": ["SPY", "TLT", "BND", "VNQ", "JNJ", "PG", "KO"],
        "₿ Crypto Portfolio": ["BTC-USD", "ETH-USD", "COIN", "MSTR", "GBTC"]
    }
    
    # Preset portfolio selection
    selected_preset = st.sidebar.selectbox(
        "🎯 Preset Portfolios",
        options=["Custom Selection"] + list(preset_portfolios.keys()),
        help="Choose a preset portfolio or select 'Custom Selection' for manual selection"
    )
    
    # Get preset assets if selected
    preset_assets = []
    if selected_preset != "Custom Selection":
        preset_assets = preset_portfolios[selected_preset]
        st.sidebar.info(f"📋 Selected: {selected_preset}")
    
    # Show quick category selection (only if custom selection)
    if selected_preset == "Custom Selection":
        selected_categories = st.sidebar.multiselect(
            "Quick Category Selection",
            options=list(asset_categories.keys()),
            help="Select entire categories to quickly add multiple assets"
        )
    else:
        selected_categories = []
    
    # Get assets from selected categories
    category_assets = []
    for category in selected_categories:
        category_assets.extend(asset_categories[category])
    
    # Remove duplicates and maintain order
    category_assets = list(dict.fromkeys(category_assets))
    
    # Asset search and selection
    st.sidebar.subheader("🔍 Asset Search & Selection")
    
    # Add search functionality
    search_term = st.sidebar.text_input(
        "Search Assets",
        placeholder="Type to filter assets (e.g., 'AAPL', 'tech', 'ETF')",
        help="Filter assets by symbol or category"
    )
    
    # Filter assets based on search
    if search_term:
        search_lower = search_term.lower()
        filtered_assets = [
            asset for asset in available_assets 
            if search_lower in asset.lower() or 
            any(search_lower in category.lower() for category in asset_categories.keys() 
                if asset in asset_categories[category])
        ]
    else:
        filtered_assets = available_assets
    
    # Show filtered count
    if search_term and filtered_assets:
        st.sidebar.info(f"🔍 Found {len(filtered_assets)} assets matching '{search_term}'")
    elif search_term and not filtered_assets:
        st.sidebar.warning(f"❌ No assets found matching '{search_term}'")
    
    # Determine default selection priority
    if preset_assets:
        default_assets = preset_assets
    elif category_assets:
        default_assets = category_assets[:4]
    else:
        default_assets = filtered_assets[:4] if len(filtered_assets) >= 4 else filtered_assets
    
    # Asset selection with filtered options
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        options=filtered_assets,
        default=default_assets,
        help="Choose individual assets, use preset portfolios, or category selection for quick picks"
    )
    
    # Show selected assets count
    if selected_assets:
        st.sidebar.success(f"✅ {len(selected_assets)} assets selected")
    else:
        st.sidebar.warning("⚠️ Please select at least one asset")
    
    # AI Strategy generation options
    st.sidebar.header("🤖 AI Strategy Generation")
    
    num_strategies = st.sidebar.slider(
        "Number of Strategies to Generate",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # AI Agent settings
    run_agent = st.sidebar.button("Generate Strategies with AI Agent")
    
    # Main content area
    st.title("PortfolioCrafter: AI-Powered Trading Strategy Platform")
    
    # Add AI Agent explanation section
    with st.expander("🤖 How Our AI Agents Work", expanded=False):
        st.markdown("""
        ### 🧠 AI Agent Architecture
        
        PortfolioCrafter uses advanced AI agents that autonomously analyze markets and generate trading strategies. Here's how they work:
        
        #### 🔄 **Agent Workflow**
        """)
        
        # Create a flowchart using markdown
        st.markdown("""
        ```mermaid
        flowchart TD
            A[📊 Market Data Input] --> B[🧠 AI Market Analyzer]
            B --> C[📈 Feature Engineering Agent]
            C --> D[🎯 Regime Detection Agent]
            D --> E[💡 Strategy Generation Agent]
            E --> F[⚡ Performance Simulation]
            F --> G[📋 Strategy Optimization]
            G --> H[🎨 Visualization & Reports]
            
            subgraph "🤖 AI Agent Layer"
                B
                C
                D
                E
            end
            
            subgraph "📊 Data Processing"
                A
                F
                G
            end
            
            subgraph "📈 Output Generation"
                H
            end
            
            classDef agent fill:#ffd700,stroke:#333,stroke-width:3px
            classDef process fill:#e1f5fe,stroke:#333,stroke-width:2px
            classDef output fill:#c8e6c9,stroke:#333,stroke-width:2px
            
            class B,C,D,E agent
            class A,F,G process
            class H output
        ```
        """)
        
        st.markdown("""
        #### 🎯 **Agent Capabilities**
        
        **1. 🧠 Market Analyzer Agent**
        - Analyzes real-time market data from multiple sources
        - Identifies market trends, volatility patterns, and correlations
        - Processes 50+ technical indicators automatically
        
        **2. 📈 Feature Engineering Agent**
        - Computes advanced technical indicators (RSI, MACD, Bollinger Bands)
        - Creates momentum, volatility, and trend features
        - Handles missing data and data quality issues
        
        **3. 🎯 Regime Detection Agent**
        - Identifies current market conditions (Bull, Bear, Sideways)
        - Detects volatility regimes (High, Medium, Low)
        - Adapts strategy parameters based on market environment
        
        **4. 💡 Strategy Generation Agent**
        - Uses Large Language Models (LLM) for intelligent strategy creation
        - Generates multiple strategy types (Momentum, Mean Reversion, etc.)
        - Creates mathematical formulations and trading rules
        - Optimizes parameters for current market conditions
        
        **5. ⚡ Performance Simulation**
        - Simulates strategy performance using historical data
        - Calculates risk metrics (Sharpe ratio, drawdown, volatility)
        - Provides realistic transaction costs and slippage
        
        **6. 📋 Strategy Optimization**
        - Automatically optimizes strategy parameters
        - Tests multiple parameter combinations
        - Selects best-performing configurations
        
        #### 🚀 **Why AI Agents?**
        
        - **⚡ Speed**: Generate strategies in minutes vs. weeks of manual work
        - **🎯 Accuracy**: AI agents process vast amounts of data without human bias
        - **🔄 Adaptability**: Strategies adapt to changing market conditions
        - **📊 Comprehensiveness**: Analyze multiple assets and strategies simultaneously
        - **🧠 Intelligence**: Use advanced reasoning to create novel strategy combinations
        """)
    
    # Add a section about the AI process
    with st.expander("🔬 AI Strategy Generation Process", expanded=False):
        st.markdown("""
        ### 🎯 **Step-by-Step AI Process**
        
        #### **Phase 1: Data Intelligence** 📊
        - **Market Data Ingestion**: Real-time data from yfinance, FRED APIs
        - **Data Quality Assurance**: Automatic cleaning and validation
        - **Feature Extraction**: 50+ technical indicators computed automatically
        
        #### **Phase 2: Market Intelligence** 🧠
        - **Regime Analysis**: AI determines current market conditions
        - **Volatility Assessment**: Identifies risk levels and market stability
        - **Correlation Analysis**: Finds relationships between assets
        
        #### **Phase 3: Strategy Intelligence** 💡
        - **LLM Reasoning**: AI agents use advanced language models to reason about markets
        - **Strategy Formulation**: Creates mathematical trading rules
        - **Parameter Optimization**: Finds optimal settings for current conditions
        
        #### **Phase 4: Performance Intelligence** ⚡
        - **Historical Simulation**: Tests strategies on past data
        - **Risk Assessment**: Calculates comprehensive risk metrics
        - **Performance Ranking**: Compares and ranks strategy effectiveness
        
        #### **Phase 5: Presentation Intelligence** 📈
        - **Visual Analytics**: Creates professional charts and graphs
        - **Mathematical Documentation**: Provides exact strategy formulas
        - **Performance Reports**: Generates comprehensive analysis reports
        """)
    
    # Initialize session state for storing generated strategies
    if "strategies" not in st.session_state:
        st.session_state.strategies = []
    
    if "strategy_results" not in st.session_state:
        st.session_state.strategy_results = {}
    
    if "current_timestamp" not in st.session_state:
        st.session_state.current_timestamp = None
    
    # Run the AI agent if requested
    if run_agent:
        if not selected_assets:
            st.error("Please select at least one asset.")
            return
        
        # Show spinner during processing
        with st.spinner("Generating strategies with AI agent..."):
            try:
                # Fetch data (ensure SPY is included for reference calculations)
                data = {}
                assets_to_fetch = list(set(selected_assets + ['SPY']))  # Include SPY if not already selected
                
                # Debug: Show which assets we're fetching
                st.info(f"Fetching data for: {', '.join(assets_to_fetch)}")
                
                for asset in assets_to_fetch:
                    try:
                        asset_data = fetch_ohlcv_data(asset, start_date, end_date)
                        if asset_data is not None and not asset_data.empty:
                            data[asset] = asset_data
                            st.success(f"✅ {asset}: {len(asset_data)} records")
                        else:
                            st.warning(f"⚠️ {asset}: No data available")
                    except Exception as e:
                        st.error(f"❌ {asset}: Error fetching data - {str(e)}")
                
                # Check if we have enough data
                if not data:
                    st.error("No data available for any selected assets. Please check your asset selection and date range.")
                    return
                
                if 'SPY' not in data:
                    st.warning("SPY data not available. Using first available asset as reference.")
                    ref_asset = list(data.keys())[0]
                else:
                    ref_asset = 'SPY'
                
                # Debug: Show data summary
                with st.expander("🔍 Data Debug Info", expanded=False):
                    st.write("**Available Data:**")
                    for asset, df in data.items():
                        st.write(f"- {asset}: {len(df)} records, columns: {list(df.columns)}")
                        if len(df) > 0:
                            st.write(f"  - Date range: {df.index[0]} to {df.index[-1]}")
                
                # Compute features using available data
                try:
                    features_df = compute_features(data, ref_asset_ticker=ref_asset)
                    st.success(f"✅ Features computed using {ref_asset} as reference")
                    
                    # Debug: Show features info
                    with st.expander("🔍 Features Debug Info", expanded=False):
                        st.write(f"**Features DataFrame:**")
                        st.write(f"- Shape: {features_df.shape}")
                        st.write(f"- Columns: {list(features_df.columns)}")
                        st.write(f"- Index type: {type(features_df.index)}")
                        st.write(f"- First few rows:")
                        st.dataframe(features_df.head())
                        
                except Exception as e:
                    st.error(f"❌ Error computing features: {str(e)}")
                    return
                
                # Detect market regime
                try:
                    regime = detect_regime(features_df)
                    st.info(f"🎯 Detected market regime: {regime}")
                except Exception as e:
                    st.warning(f"⚠️ Could not detect market regime: {str(e)}. Using default regime.")
                    regime = "Unknown"
                
                # Run a baseline momentum strategy for comparison
                baseline_params = {"fast_window": 21, "slow_window": 63}
                try:
                    baseline_result = run_backtest(
                        data,
                        [selected_assets[0]],
                        "momentum",
                        baseline_params
                    )
                    # Make robust to different return types (dict/Series/None)
                    if isinstance(baseline_result, dict):
                        baseline_stats = pd.Series(baseline_result.get("metrics", {}))
                    elif isinstance(baseline_result, pd.Series):
                        baseline_stats = baseline_result
                    else:
                        baseline_stats = pd.Series({})
                    st.success("✅ Baseline strategy executed successfully")
                except Exception as e:
                    st.warning(f"⚠️ Baseline strategy failed: {str(e)}. Using empty baseline.")
                    baseline_stats = pd.Series({})
                
                # Normalize regime to a dict payload for downstream compatibility
                if isinstance(regime, str):
                    if 'HighVol' in regime or 'Crisis' in regime:
                        est_vol = 0.25
                    elif 'MidVol' in regime:
                        est_vol = 0.18
                    else:
                        est_vol = 0.12
                    regime_payload = {
                        'name': regime,
                        'current_regime': regime,
                        'current_volatility': est_vol
                    }
                elif isinstance(regime, dict):
                    regime_payload = regime
                else:
                    regime_payload = {'name': str(regime)}

                # Generate strategies with the AI agent
                try:
                    proposals = generate_strategy_proposals(
                        regime_data=regime_payload,
                        features_df=features_df,
                        baseline_stats=baseline_stats,
                        strategy_types=available_strategies,
                        available_assets=selected_assets,
                        num_proposals=num_strategies
                    )
                    st.success(f"✅ Generated {len(proposals)} AI strategies")
                except Exception as e:
                    st.error(f"❌ Error generating strategies: {str(e)}")
                    return
                
                # Store the generated strategies
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.current_timestamp = timestamp
                st.session_state.strategies = proposals
                
                # Create folder for saving results
                results_folder = get_timestamp_folder()
                
                # Run performance simulation for each AI-generated strategy
                strategy_results = {}
                for i, strategy in enumerate(proposals):
                    try:
                        # Sanitize strategy params: rename threshold->threshold_pct, remove stray window keys
                        params = strategy.get("params", {}).copy()
                        if 'threshold' in params and 'threshold_pct' not in params:
                            params['threshold_pct'] = params.pop('threshold')
                        if strategy.get("strategy_type") in ['trend_following', 'regime_based'] and 'window' in params:
                            params.pop('window')
                        result = run_backtest(
                            data,
                            strategy["asset_tickers"],
                            strategy["strategy_type"],
                            params,
                            strategy.get("allocation_weights")
                        )
                        
                        # Store the result if performance simulation succeeded
                        strategy_key = f"Strategy {i+1}: {strategy['strategy_type']}"
                        if result is not None:
                            strategy_results[strategy_key] = {
                                "strategy_info": strategy,
                                "result": result,
                                "equity_curve": result.get("equity_curve"),
                                "weights": result.get("weights"),
                                "metrics": result.get("metrics")
                            }
                            
                            # Create dashboard plots
                            try:
                                # Get benchmark data safely
                                benchmark_data = None
                                try:
                                    if selected_assets[0] in data:
                                        benchmark_df = data[selected_assets[0]]
                                        if 'Close' in benchmark_df.columns:
                                            benchmark_data = benchmark_df["Close"]
                                        elif 'close' in benchmark_df.columns:
                                            benchmark_data = benchmark_df["close"]
                                        elif 'Adj Close' in benchmark_df.columns:
                                            benchmark_data = benchmark_df["Adj Close"]
                                except Exception:
                                    benchmark_data = None
                                
                                create_strategy_dashboard(
                                    equity_curve=result.get("equity_curve"),
                                    weights_df=result.get("weights"),
                                    strategy_info=strategy,
                                    benchmark=benchmark_data,
                                    save_path=results_folder
                                )
                            except Exception as plot_error:
                                st.warning(f"Failed to create plots for strategy {i+1}: {plot_error}")
                        else:
                            st.error(f"Performance simulation failed for strategy {i+1}: {strategy['strategy_type']}")
                        
                    except Exception as e:
                        st.error(f"Error running performance simulation for strategy {i+1}: {str(e)}")
                
                # Store the strategy results
                st.session_state.strategy_results = strategy_results
                
                st.success(f"Generated {len(proposals)} strategies and saved results to {results_folder}")
            
            except Exception as e:
                st.error(f"Error generating strategies: {str(e)}")
    
    # Display the generated strategies and results
    if st.session_state.strategies:
        st.header("Generated Trading Strategies")
        
        # Create tabs for the strategies
        strategy_tabs = st.tabs([f"Strategy {i+1}: {s['strategy_type']}" for i, s in enumerate(st.session_state.strategies)])
        
        for i, (tab, strategy) in enumerate(zip(strategy_tabs, st.session_state.strategies)):
            strategy_key = f"Strategy {i+1}: {strategy['strategy_type']}"
            
            with tab:
                # Display strategy details
                st.subheader(f"{strategy['strategy_type'].title()} Strategy")
                st.write(f"**Rationale**: {strategy['rationale']}")
                
                # Display parameters
                st.write("**Parameters:**")
                for param, value in strategy['params'].items():
                    st.write(f"- {param}: {value}")
                
                # Display asset allocation
                st.write("**Asset Allocation:**")
                if strategy.get('allocation_weights'):
                    allocation_df = pd.DataFrame([strategy['allocation_weights']])
                    st.dataframe(allocation_df)
                else:
                    st.write("Equal weighting across assets")
                
                # Display strategy results if available
                if strategy_key in st.session_state.strategy_results and st.session_state.strategy_results[strategy_key]["result"] is not None:
                    result_data = st.session_state.strategy_results[strategy_key]
                    
                    # Display metrics
                    st.subheader("Performance Metrics")
                    metrics_df = pd.DataFrame([result_data["metrics"]])
                    st.dataframe(metrics_df)
                    
                    # Create columns for the plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plot equity curve
                        st.subheader("Portfolio Performance")
                        fig = plot_portfolio_performance(
                            result_data["equity_curve"],
                            benchmark=None  # We could add a benchmark here
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        # Plot allocation weights
                        st.subheader("Asset Allocation")
                        weights_data = result_data["weights"]
                        if isinstance(weights_data, dict):
                            # Convert static weights dict to DataFrame for display
                            weights_df = pd.DataFrame([weights_data])
                            st.dataframe(weights_df)
                            # Also create a simple pie chart
                            fig, ax = plt.subplots(figsize=(8, 6))
                            assets = list(weights_data.keys())
                            weights = list(weights_data.values())
                            ax.pie(weights, labels=assets, autopct='%1.1f%%')
                            ax.set_title("Asset Allocation")
                            st.pyplot(fig)
                            plt.close(fig)
                        elif isinstance(weights_data, pd.DataFrame):
                            fig = plot_portfolio_composition(weights_data)
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.write("No allocation data available")
                    
                    # Plot strategy formula
                    st.subheader("Strategy Formula")
                    fig = plot_strategy_formula(strategy)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Add AI parameter optimization option
                    st.subheader("🤖 AI Parameter Optimization")
                    if st.button("Optimize with AI", key=f"optimize_{i}"):
                        with st.spinner("AI is optimizing parameters..."):
                            try:
                                # Get the data for optimization (ensure SPY is included for reference)
                                data = {}
                                assets_to_fetch = list(set(strategy["asset_tickers"] + ['SPY']))
                                for asset in assets_to_fetch:
                                    data[asset] = fetch_ohlcv_data(asset, start_date, end_date)
                                
                                # Run AI optimization
                                opt_result = optimize_strategy_parameters(
                                    strategy,
                                    data,
                                    num_trials=30
                                )
                                
                                # Display optimization results
                                st.success("🎉 AI optimization complete!")
                                
                                # Show optimized parameters
                                st.write("**🤖 AI-Optimized Parameters:**")
                                for param, value in opt_result["optimized_params"].items():
                                    st.write(f"- {param}: {value}")
                                
                                st.write(f"**📈 Best Performance Score:** {opt_result['best_sharpe']:.4f}")
                                
                                # Option to apply optimized parameters
                                if st.button("Apply AI-Optimized Parameters", key=f"apply_opt_{i}"):
                                    # Update strategy with AI-optimized parameters
                                    strategy["params"] = opt_result["optimized_params"]
                                    
                                    # Re-run performance simulation with optimized parameters
                                    result = run_backtest(
                                        data,
                                        strategy["asset_tickers"],
                                        strategy["strategy_type"],
                                        strategy["params"],
                                        strategy.get("allocation_weights")
                                    )
                                    
                                    # Update stored results
                                    st.session_state.strategy_results[strategy_key]["result"] = result
                                    st.session_state.strategy_results[strategy_key]["equity_curve"] = result.get("equity_curve")
                                    st.session_state.strategy_results[strategy_key]["weights"] = result.get("weights")
                                    st.session_state.strategy_results[strategy_key]["metrics"] = result.get("metrics")
                                    
                                    # Rerun the app to show updated results
                                    st.experimental_rerun()
                            
                            except Exception as e:
                                st.error(f"Error during AI optimization: {str(e)}")
                else:
                    st.warning("No performance simulation results available for this strategy.")
    
    else:
        st.info("Click 'Generate Strategies with AI Agent' to get started.")


if __name__ == "__main__":
    main()
