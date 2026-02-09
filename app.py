import sys
import warnings
import plotly.express as px
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import statsmodels.api as sm


st.set_page_config(page_title="Portfolio Analysis", layout="wide")

# data pipeline
STATIC_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'MA', 
    'JPM', 'UNH', 'JNJ', 'XOM', 'AVGO', 'HD', 'PG', 'COST', 'ADBE', 'LLY', 
    'CVX', 'MRK', 'ABBV', 'KO', 'PEP', 'BAC', 'WMT', 'TMO', 'CRM', 'MCD', 
    'ACN', 'CSCO', 'NFLX', 'ORCL', 'ABT', 'DIS', 'DHR', 'INTC', 'VZ', 'TXN', 
    'AMAT', 'PM', 'PFE', 'LOW', 'UNP', 'SPY', 'QQQ', 'VOO', 'IWM', 'DIA', 'GLD'
]

@st.cache_data
def get_stock_data(tickers, start_date):
    if not tickers:
        return None
    try:
        data = yf.download(tickers, start=start_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                stock_data = data['Close']
            except KeyError:
                stock_data = data.xs('Close', level=1, axis=1)  # this meant to bypass version changes of yfinance
        else:
            stock_data = data['Close'] if 'Close' in data else data
        
        if stock_data is None or stock_data.empty:
            return None
            
        return stock_data.dropna()
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None
#

#functions: mathematical engine
        """
        when we multiply by 252, we do it to annualize
        on avg, there are 252 trading days a year on most markets
        so crypto and markets with other measurements will be off.
        the app doesnt handle exchange rate difference between currencies
        """
def perform_analysis(stock_data):
    log_returns = np.log(stock_data / stock_data.shift(1))
    log_returns = log_returns.dropna()
    #log returns are used for time series analysis: they are additive + follow normal distribution more closely
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    tickers = stock_data.columns.tolist()
    return log_returns, mean_returns, cov_matrix, tickers

def run_monte_carlo(mean_returns, cov_matrix, num_simulations, risk_free_rate):
    num_assets = len(mean_returns)
    weights = np.random.random((num_simulations, num_assets))  # creates a matrix filled with numbers [0;1] =weights
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis] # now the sum of weights in each row will be 1
                                                                # np.newaxis broadcasts sums (a list) into a (m,1) matrix
    port_returns = np.dot(weights, mean_returns) # E(R_p) = w_1*r_1+w_2*r_2+...+w_n*r_n    =weighted mean portfolio return
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
    # i: simulation index, j/k: asset index. calculates portfolio volatility for all simulations. portfolio volatility = square root of (transposed weights * covariance * weights). though, np.einsum automatically transposes
    sharpe_ratios = (port_returns - risk_free_rate) / port_vols
    return weights, port_returns, port_vols, sharpe_ratios

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, objective='sharpe'):
    # returns the optimal portfolio weights for selected objective (max sharpe or min vol.Ö
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # sum of all weights must equal 1. fun(x)=0 -> sum(x)-1=0
    bounds = tuple((0, 1) for asset in range(num_assets)) # weights must be between 0-1 for each asset
    init_guess = num_assets * [1.0 / num_assets,] # initial guess with an equal distribution helps converge fsater
    
    if objective == 'sharpe': # objective is to maximize sharpe ratio
                                # since max(f(x))=min(-f(x)), we minimize the negative sharpe ratio
        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
            p_ret = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - (p_ret - risk_free_rate) / p_vol
        result = minimize(neg_sharpe, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                                                                # SLSQP = sequential least squares programming
    elif objective == 'volatility': # portfolio with minimum volatility
        def port_vol(weights, mean_returns, cov_matrix, rf):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        result = minimize(port_vol, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=50):
    # draws the efficient frontier, by minimizing volatility for fixed levels of returns
    num_assets = len(mean_returns) # define the range of y-axis: min-max return
    min_ret = min(mean_returns)
    max_ret = max(mean_returns)
    target_returns = np.linspace(min_ret, max_ret, num_points)
    efficient_volatility = []
    bounds = tuple((0, 1) for asset in range(num_assets)) 
    init_guess = num_assets * [1.0 / num_assets,]
    
    for target in target_returns: # for every target return level, find the minimum risk (x)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # weights sum = 1
            {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target} # lock returns and only minimize variance
        )
        
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                          init_guess, method='SLSQP', bounds=bounds, constraints=constraints) #minimize volatility
        
        efficient_volatility.append(result.fun if result.success else np.nan) # save x coordinates if succeed, otherwise nan to avoid plotting error
    return efficient_volatility, target_returns

def run_future_simulation(weights, mean_returns, cov_matrix, years, num_simulations=1000):
    """
    simulates future portfolio paths using Geometric Brownian Motion
    S_t = S_0 * exp(((mu- 0,5sigma^2)*t + sigma*W_t)
    (W_t is a Wiener process)
    """
    port_mu = np.sum(mean_returns * weights) # expected annual return
    port_sigma = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) # portfolio volatility
    dt = 1/252 # delta time, one time step given in years
    num_days = int(years * 252) # total number of time steps
    random_shocks = np.random.normal(0, 1, (num_days, num_simulations)) #standard normal random variables for Weiner process
    drift = (port_mu - 0.5 * port_sigma**2) * dt
    diffusion = port_sigma * np.sqrt(dt) * random_shocks
    daily_log_returns = drift + diffusion
    cumulative_returns = np.exp(np.cumsum(daily_log_returns, axis=0)) #convert log->cummulative 
    portfolio_paths = np.vstack([np.ones((1, num_simulations)), cumulative_returns]) # prepend 1.0 initial investment to t=0 for every simulation
    return portfolio_paths

def clean_weights(weights, tickers, threshold=0.0001):
    # optimizes results to remove noise (assets with weight<threshold), then re-normalizes
    df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
    df['Weight'] = df['Weight'].apply(lambda x: x if x > threshold else 0.0)
    if df['Weight'].sum() > 0:
        df['Weight'] = df['Weight'] / df['Weight'].sum()
    return df.sort_values(by='Weight', ascending=False)

def calculate_risk_contribution(weights, cov_matrix):
    """
    returns an array which contains what percentage of the total risk comes from each asset
    """
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # marginal contribution to risk: partial derivative with respect to the weights (simplified)
    # shows the sensitivity of risk if weights are changed
    mcr = np.dot(cov_matrix, weights) / port_vol
    # component risk controbution: absolute amount of volatility contributed by each asset
    rc = weights * mcr
    
    rc_percent = rc / port_vol # normalize
    return rc_percent

def calculate_cvar(returns, confidence=0.95): # conditional value at risk: average loss in the worst 5% of cases
    var_threshold = returns.quantile(1 - confidence)
    cvar = returns[returns <= var_threshold].mean()
    return cvar


# UI, options

st.sidebar.header("Options")

if 'results' not in st.session_state:
    st.session_state.results = None

selected_tickers = st.sidebar.multiselect(
    "Select Tickers:",
    STATIC_TICKERS, 
    default=['JPM','GLD','GOOGL','CVX'],
)

custom_input = st.sidebar.text_input("Comma separated custom tickers (e.g. MOL.BD, OTP.BD):")
if custom_input:
    selected_tickers.extend([x.strip().upper() for x in custom_input.split(',')])
    selected_tickers = list(set(selected_tickers))

start_date = st.sidebar.date_input("Start date:", value=pd.to_datetime("2019-01-01"))
risk_free_rate = st.sidebar.number_input("Risk free rate:", value=0.04, step=0.01, help="0,04 means 4%")
num_sims = st.sidebar.number_input("Number of simulations:",100,20000,1407,100,help="Higher number = more accuracy, but might cause performance issues")

run_btn = st.sidebar.button("Run Analysis", type="primary")


if run_btn:
    if not selected_tickers:
        st.warning("Please select at least one ticker.")
    else:
        with st.spinner('Calculating Mathematical Optimum...'):
            stock_data = get_stock_data(selected_tickers, start_date)
            benchmark_data = get_stock_data(['SPY'], start_date)
            
            # iff benchmark_data comes back as DataFrame, ensure we just get the Close series
            if isinstance(benchmark_data, pd.DataFrame) and not benchmark_data.empty:
                 if benchmark_data.shape[1] > 0:
                    benchmark_data = benchmark_data.iloc[:, 0]

            if stock_data is not None and not stock_data.empty:   # run calculations
                log_returns, mu, S, tickers = perform_analysis(stock_data)
                max_sharpe_res = optimize_portfolio(mu, S, risk_free_rate, 'sharpe')
                min_vol_res = optimize_portfolio(mu, S, risk_free_rate, 'volatility')
                eff_vols, eff_rets = calculate_efficient_frontier(mu, S, risk_free_rate)
                mc_weights, mc_rets, mc_vols, mc_sharpes = run_monte_carlo(mu, S, num_sims, risk_free_rate)
                
                st.session_state.results = { # had issues while changing options during predictions, so we use session
                    'stock_data': stock_data,
                    'benchmark_data': benchmark_data,
                    'mu': mu, 'S': S, 'tickers': tickers,
                    'mc': (mc_weights, mc_rets, mc_vols, mc_sharpes),
                    'sharpe_res': max_sharpe_res,
                    'vol_res': min_vol_res,
                    'frontier': (eff_vols, eff_rets),
                    'log_returns': log_returns
                }
            else:
                st.error("Failed to download data.")

# dashboard

if st.session_state.results is not None:
    res = st.session_state.results
    mu, S, tickers = res['mu'], res['S'], res['tickers']
    mc_weights, mc_rets, mc_vols, mc_sharpes = res['mc']
    max_sharpe_res, min_vol_res = res['sharpe_res'], res['vol_res']
    eff_vols, eff_rets = res['frontier']
    stock_data = res['stock_data']
    log_returns = res['log_returns']
    benchmark_data = res['benchmark_data'] # Unpack benchmark

    # extract optimal values from the solver, Sharpe
    opt_w_sharpe = max_sharpe_res.x
    opt_ret_sharpe = np.sum(mu * opt_w_sharpe)
    opt_vol_sharpe = np.sqrt(np.dot(opt_w_sharpe.T, np.dot(S, opt_w_sharpe)))
    opt_sharpe_ratio = (opt_ret_sharpe - risk_free_rate) / opt_vol_sharpe
    # volatility
    opt_w_vol = min_vol_res.x
    opt_ret_vol = np.sum(mu * opt_w_vol)
    opt_vol_vol = np.sqrt(np.dot(opt_w_vol.T, np.dot(S, opt_w_vol)))
    opt_sharpe_ratio_vol = (opt_ret_vol - risk_free_rate) / opt_vol_vol
    
    df_sharpe = clean_weights(opt_w_sharpe, tickers)
    df_vol = clean_weights(opt_w_vol, tickers)

    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Efficient Frontier", "Weights", "Backtest & Correlation Matrix", 
        "Prediction", "Risk metrics", "Stress Test", "Risk Budgeting"
    ])

    
    with tab1:
        st.subheader("Efficient Frontier")
        st.info("Visualizing the Modern Portfolio Theory. The red line - calculated via sequential least squares programming - represents portfolios that offer the highest expected return for a defined level of risk. A cloud of potential portfolios was generated with Monte Carlo simulation to demonstrate that the Efficient Frontier provides the optimal reward for every level of risk.")
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=mc_vols, y=mc_rets, mode='markers', marker=dict(color=mc_sharpes, colorscale='Viridis', size=3, showscale=True, colorbar=dict(title="Sharpe")), name='Simulations', opacity=0.3))
        fig.add_trace(go.Scattergl(x=eff_vols, y=eff_rets, mode='lines', line=dict(color='red', width=3, dash='dash'), opacity=0.5, name='Efficient Frontier'))
        fig.add_trace(go.Scattergl(x=[opt_vol_sharpe], y=[opt_ret_sharpe], mode='markers', marker=dict(color='red', size=10, symbol='circle-dot', line=dict(width=1, color='red')), name=f'Max Sharpe Portfolio: {opt_sharpe_ratio:.2f}'))
        fig.update_layout(xaxis_title="Volatility", yaxis_title="Expected Annual Return", height=600,
                legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01)
                          )
        st.plotly_chart(fig, width="stretch")
        st.warning("Note that the number of simulations is reduced to improve performance. More simulations mean higher accuracy over the Efficient Frontier, but it can be demanding on the system.")

    with tab2:
        st.info(r"""Using SQLSP optimization algorithm, we are able to calculate the weights of a portfolio with maximum Sharpe ratio or minimum volatility.  
        Max Sharpe maximizes the risk-adjusted return ratio, identifying the most efficient portfolio.  
        Min volatility minimizes total portfolio variance, priotizing stability.""")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Max Sharpe Portfolio")
            st.markdown(f"**Sharpe Ratio:** {opt_sharpe_ratio:.3f}")
            st.markdown(f"**Return:** {opt_ret_sharpe*100:.3f}%")
            st.markdown(f"**Volatility:** {opt_vol_sharpe*100:.3f}%")
            
            fig_pie1 = go.Figure(data=[go.Pie(
                labels=df_sharpe[df_sharpe['Weight'] > 0]['Ticker'], 
                values=df_sharpe[df_sharpe['Weight'] > 0]['Weight'],
                hole=.3,
                textinfo='label+percent'
            )])
            st.plotly_chart(fig_pie1, width="stretch", key="pie-sharpe")
            
            st.dataframe(
                df_sharpe[df_sharpe['Weight'] > 0],
                column_config={"Ticker": "Ticker", "Weight": st.column_config.NumberColumn("Weight", format="%.3f")},
                hide_index=True, use_container_width=True
            )

        with col2:
            st.markdown("### Minimum Volatility Portfolio")
            st.markdown(f"**Sharpe Ratio:** {opt_sharpe_ratio_vol:.3f}")
            st.markdown(f"**Return:** {opt_ret_vol*100:.3f}%")
            st.markdown(f"**Volatility:** {opt_vol_vol*100:.3f}%")
            
            
            fig_pie2 = go.Figure(data=[go.Pie(
                labels=df_vol[df_vol['Weight'] > 0]['Ticker'], 
                values=df_vol[df_vol['Weight'] > 0]['Weight'],
                hole=.3,
                textinfo='label+percent'
            )])
            st.plotly_chart(fig_pie2, width="stretch", key="pie-vol")
            
            # data table
            st.dataframe(
                df_vol[df_vol['Weight'] > 0],
                column_config={"Ticker": "Ticker", "Weight": st.column_config.NumberColumn("Weight", format="%.3f")},
                hide_index=True, use_container_width=True
            )


    with tab3:
        st.subheader("Backtest Performance")
        st.info("This section reconstructs the portfolio's performance using historical data with the calculated optimal weights. The graph show cumulative returns, like if you invested 1$ at the start.")
        daily_returns_all = stock_data.pct_change().dropna()
        portfolio_daily_ret = (daily_returns_all * opt_w_sharpe).sum(axis=1)
        cum_ret_portfolio = (1 + portfolio_daily_ret).cumprod() 
        cvar = calculate_cvar(portfolio_daily_ret, 0.95)
        
        fig_bt = go.Figure() # portfolio return
        fig_bt.add_trace(go.Scattergl(x=cum_ret_portfolio.index, y=cum_ret_portfolio, mode='lines', name='Optimal Portfolio', line=dict(color='red', width=3)))
        
        for ticker in tickers[:3]: # top3 individual asset
             asset_ret = (1 + daily_returns_all[ticker]).cumprod()
             fig_bt.add_trace(go.Scattergl(x=asset_ret.index, y=asset_ret, mode='lines', name=ticker, opacity=0.3))

        fig_bt.update_layout(yaxis_title="Growth Multiplier (1.0 = Start)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_bt, width="stretch")
    
        daily_vol = opt_vol_sharpe / np.sqrt(252)
        # 1,645 is the Z-score for the 95th percentile of normal distribution. calculates value at risk at 95% confidence
        var_95_daily = daily_vol * 1.645

        col1, col2 = st.columns(2)
        
        with col1:
            st.warning(f"**Daily Value at Risk (95%): {var_95_daily:.2%}** \n\n This metric is meant to estimate the maximum loss over a 1-day period with 95% confidence assuming normal distribution.")

        with col2:
            st.warning(f"**Conditional Value at Risk (95%):  {abs(cvar):.2%}** \n\n Average loss in worst 5% of days.")
            
        st.markdown("---")
        st.info("Correlation heatmap visualizes linear relationship between assets. Negative values mean anti-correlation, 0 means no correlation and values near 0 mean low correlation, which is benefical for diversification.")
        # correlation heatmap
        corr_matrix = log_returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_corr, width="stretch")

        normal_periods = portfolio_daily_ret[portfolio_daily_ret > portfolio_daily_ret.quantile(0.5)]
        stress_periods = portfolio_daily_ret[portfolio_daily_ret < portfolio_daily_ret.quantile(0.1)]
        
        corr_normal = log_returns.loc[normal_periods.index].corr()
        corr_stress = log_returns.loc[stress_periods.index].corr()

        st.info("Below, the graph shows correlation under normal conditions. (Only days with top 50% returns count)")
    #NORMAL
        fig_corr_normal = px.imshow(corr_normal, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_corr_normal, width="stretch")
        st.info("Below, the graph shows correlation under stress. (Only days with bottom 10% returns count")
    #STRESS
        fig_corr_stress = px.imshow(corr_stress, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_corr_stress, width="stretch")
        

    with tab4:
            
        st.subheader("Future Growth Projection")
        st.info("Simulates thousands of possible price paths for the portfolio using Geometric Brownian Motion. Asset prices are influenced by two components: a deterministic drift, which is the expected long term trend based on historical data, and a stochastic diffusion, which is the random volatility shock.")
        
        years_input = st.slider("Years to forecast", 1, 15, 5)
        sim_paths = run_future_simulation(opt_w_sharpe, mu, S, years_input)
        
        final_values = sim_paths[-1, :] 
        median_result = np.median(final_values)
        top_10 = np.percentile(final_values, 90)
        bottom_10 = np.percentile(final_values, 10)

        x_axis = np.arange(sim_paths.shape[0])
        upper_bound = np.percentile(sim_paths, 90, axis=1)
        lower_bound = np.percentile(sim_paths, 10, axis=1)
        median_path = np.median(sim_paths, axis=1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected", f"{median_result:.2f}x")
        c2.metric("10th percentile", f"{bottom_10:.2f}x")
        c3.metric("90th percentile", f"{top_10:.2f}x")
        
        fig_mc = go.Figure()

        # upper bound
        fig_mc.add_trace(go.Scatter(
            x=x_axis, y=upper_bound,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))

        # lower bound
        fig_mc.add_trace(go.Scatter(
            x=x_axis, y=lower_bound,
            mode='lines', line=dict(width=0),
            fill='tonexty',  
            fillcolor='rgba(0, 100, 255, 0.2)', 
            showlegend=False, hoverinfo='skip',
            name='Confidence Interval'
        ))

        # random simulations
        for i in range(min(100, sim_paths.shape[1])): 
            fig_mc.add_trace(go.Scattergl(x=x_axis, y=sim_paths[:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))
        
            # median
        fig_mc.add_trace(go.Scattergl(x=x_axis, y=median_path, mode='lines', line=dict(color='blue', width=3), name='Median'))

        fig_mc.update_layout(title=f"Growth Projection", xaxis_title="Trading Days", yaxis_title="Multiplier")
        st.plotly_chart(fig_mc, width="stretch")

        st.info(r"""
        Stochastic Differential Equation:
        
        $$
        dS_t = \mu S_t dt + \sigma S_t dW_t
        $$
        
        *Where $dS_t$ is the change in price, $\mu$ is the expected return, $\sigma$ is volatility, and $dW_t$ is random market noise (Wiener-process).*
        """)
        st.warning(r"""Assumptions which may not hold to reality:  
        - Returns follow log-normal distribution  
        - Volatility remains constant  
        - Historical statistics predict future behaviour  
        Real portfolio results will differ due to fat tails, structural breaks and black swan events.""")

    with tab5:
        st.info(r"""Drawdown graph measures percentage drop from the portfolio's peak of the inspected timeframe.  
        Beta Regression plot shows market sensitivity. It compares the daily returns of our portfolio against S&P 500's.  
        The red line is the linear regression model. Its' slope represents the portfolio's Beta. A steep line (Beta>1) means aggressive, a flat line (Beta<1) means defensive portfolio.  
        Alpha is where the line crosses the vertical axis. If Alpha>0, the portfolio generates profit when the market is flat. If it's below zero, the portfolio underperforms.
        """)
        if benchmark_data is not None and not benchmark_data.empty:
            
            bench_ret = benchmark_data.pct_change().dropna() # daily percentage change for SPY
            
            if isinstance(bench_ret, pd.DataFrame): # ensure bench_ret is a series 1D array
                 bench_ret = bench_ret.iloc[:, 0]

                # ensure we only use dates which exist both in SPY and benchmark data. important for regression
            common_idx = portfolio_daily_ret.index.intersection(bench_ret.index)
            port_ret_aligned = portfolio_daily_ret.loc[common_idx]
            bench_ret_aligned = bench_ret.loc[common_idx]
            # we add constant X to calculate alpha. without this, the line would be forced through (0,0).
            X = sm.add_constant(bench_ret_aligned)
            model = sm.OLS(port_ret_aligned, X).fit() # ordinary least squares regression model
            
            alpha_daily = model.params['const']
            alpha_annual = alpha_daily * 252 
            beta_val = model.params.iloc[1] # slope
            r_squared = model.rsquared 
            
            downside_returns = port_ret_aligned[port_ret_aligned < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (opt_ret_sharpe - risk_free_rate) / downside_std
            skewness = portfolio_daily_ret.skew()
            kurtosis = portfolio_daily_ret.kurtosis()

            cum_ret = (1 + portfolio_daily_ret).cumprod()
            running_max = cum_ret.cummax()
            drawdown = (cum_ret / running_max) - 1 # percentage drop from local max
            max_drawdown = drawdown.min()

            rolling_sharpe = ((portfolio_daily_ret.rolling(window=252, min_periods=60).mean() * 252) / \
                 (portfolio_daily_ret.rolling(window=252, min_periods=60).std() * np.sqrt(252))).dropna()
            
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1: # drawdown plot
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', mode='lines', line=dict(color='red', width=1), name='Drawdown'))
                fig_dd.update_layout(yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, width="stretch")
                
            with col_graph2: # CAPM regression
                fig_capm = go.Figure()
                fig_capm.add_trace(go.Scatter(x=bench_ret_aligned, y=port_ret_aligned, mode='markers', name='Daily Returns', marker=dict(opacity=0.5)))
                # plotting the regression trendline
                x_range = np.linspace(min(bench_ret_aligned), max(bench_ret_aligned), 100)
                y_pred = alpha_daily + beta_val * x_range
                fig_capm.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Trendline', line=dict(color='red')))
                fig_capm.update_layout(title=f"Beta Regression (Beta: {beta_val:.2f})", xaxis_title="SPY", yaxis_title="Portfolio")
                st.plotly_chart(fig_capm, width="stretch")

            fig_roll_s = go.Figure() # rolling sharpe
            fig_roll_s.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', name='Rolling Sharpe', line=dict(color='red')))
            fig_roll_s.add_hline(y=1, line_dash="dot")
            fig_roll_s.update_layout(yaxis_title="Sharpe")
            st.plotly_chart(fig_roll_s, width="stretch")
            st.info("Rolling Sharpe Ratio measures risk-adjusted performance over time using a 252-day window")

            col_adv1, col_adv2, col_adv3 = st.columns(3)
            
            with col_adv1:
                st.metric("Beta", f"{beta_val:.2f}", help="Sensitivity")
                st.metric("Alpha", f"{alpha_annual:.2%}", help="Annualized excess return")
                st.metric("R-Squared", f"{r_squared:.2f}")

            with col_adv2:
                st.metric("Sharpe Ratio", f"{opt_sharpe_ratio:.2f}")
                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}", help="Downside risk adjusted")
                st.metric("Skewness", f"{skewness:.2f}", 
          help="Negative = more left tail risk. Symmetric = 0")
            
            with col_adv3:
                st.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")
                st.metric("Current Drawdown", f"{drawdown.iloc[-1]:.2%}", delta_color="inverse")
                st.metric("Excess Kurtosis", f"{kurtosis:.2f}", 
          help="Fat tails if > 3. Normal distribution = 3")
                
        else:
             st.warning("Benchmark data not available.")

        st.info(r""" $R^2$ measures the correlation reliability between the portfolio and the benchmark.  
        Beta is only a reliable measure of risk if $R^2$ is reasonably high.  
        Sortino ratio is similar to Sharpe, but it only measures downside volatility.""")

    with tab6:
        st.subheader("Historical Stress Testing")
        
        scenarios = {
            "2022 Tech Bear Market": ("2022-01-01", "2022-12-31"),
            "2020 COVID Crash": ("2020-02-19", "2020-03-23"),
            "Recent 3 Months": (str(stock_data.index[-65].date()), str(stock_data.index[-1].date()))
        }
        
        selected_scenario = st.selectbox("Select a Crisis Scenario:", list(scenarios.keys()))
        start_scen, end_scen = scenarios[selected_scenario]
        
        try:
            if benchmark_data is not None and not benchmark_data.empty:
                # slice user portfolio to the relevant interval
                subset_data = stock_data.loc[start_scen:end_scen]
                
                if not subset_data.empty:
                    subset_ret = subset_data.pct_change().dropna()
                    subset_port_ret = (subset_ret * opt_w_sharpe).sum(axis=1) # apply current weights to historical returns
                    subset_cum_ret = (1 + subset_port_ret).cumprod()
                    
                    bench_subset = benchmark_data.loc[start_scen:end_scen].pct_change().dropna() # slice SPY data the same
                    if isinstance(bench_subset, pd.DataFrame):
                        bench_subset = bench_subset.iloc[:, 0]
                        
                    bench_cum_ret = (1 + bench_subset).cumprod()

                    if not subset_cum_ret.empty and not bench_cum_ret.empty:
                        total_return = float(subset_cum_ret.iloc[-1] - 1)
                        bench_return = float(bench_cum_ret.iloc[-1] - 1)
                        
                        col_stress1, col_stress2, col_stress3 = st.columns(3)
                        col_stress1.metric("Portfolio Return", f"{total_return:.2%}")
                        col_stress2.metric("Market (SPY) Return", f"{bench_return:.2%}")
                        col_stress3.metric("Alpha (Excess)", f"{total_return - bench_return:.2%}")
                        
                        fig_stress = go.Figure()
                        fig_stress.add_trace(go.Scatter(x=subset_cum_ret.index, y=subset_cum_ret, mode='lines', name='Portfolio', line=dict(color='red', width=3)))
                        fig_stress.add_trace(go.Scatter(x=bench_cum_ret.index, y=bench_cum_ret, mode='lines', name='S&P 500', line=dict(color='gray', dash='dot')))
                        fig_stress.update_layout(title=f"Performance during {selected_scenario}", yaxis_title="Growth")
                        st.plotly_chart(fig_stress, width="stretch")
                    else:
                        st.warning("Not enough data points for this scenario.")
                else:
                    st.warning("No data available for this scenario (dates might be out of range).")
            else:
                 st.warning("Benchmark data missing.")
        except Exception as e:
            st.error(f"Could not calculate scenario: {e}")

        st.markdown("---")
        st.subheader("Rolling Beta (60-day)") # rolling beta shows how sensitivity changes over time
        
        try:
            if benchmark_data is not None and not benchmark_data.empty:
                window = 60 
                
                bench_ret_roll = benchmark_data.pct_change().dropna()
                if isinstance(bench_ret_roll, pd.DataFrame):
                    bench_ret_roll = bench_ret_roll.iloc[:, 0]

                portfolio_daily_ret = (stock_data.pct_change().dropna() * opt_w_sharpe).sum(axis=1)
                # data alignment
                rolling_common = portfolio_daily_ret.index.intersection(bench_ret_roll.index)
                y_roll = portfolio_daily_ret.loc[rolling_common]
                x_roll = bench_ret_roll.loc[rolling_common]
                # beta = cov(asset, market) / var(market)
                rolling_cov = y_roll.rolling(window).cov(x_roll)
                rolling_var = x_roll.rolling(window).var()
                rolling_beta = rolling_cov / rolling_var
                
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode='lines', name='Rolling Beta', line=dict(color='blue')))
                fig_roll.add_hline(y=1, line_dash="dot")
                fig_roll.update_layout(yaxis_title="Beta")
                st.plotly_chart(fig_roll, width="stretch")
        except Exception as e:
            st.warning(f"Could not calculate rolling beta: {e}")

        st.info(r"""
        - Consistent beta suggests predictable market exposure   
        - Sharp spikes indicate the portfolio became riskier during market stress   
        - Beta dropping toward 0 or negative during crashes shows downside protection""")

    with tab7:
        st.info("Comparing Capital Allocation vs. Risk Allocation. Usually, risky assets dominate the risk budget even with small weights.")
        
        risk_contribution = calculate_risk_contribution(opt_w_sharpe, S)
        
        df_risk = pd.DataFrame({
            "Ticker": tickers,
            "Capital Weight": opt_w_sharpe,
            "Risk Contribution": risk_contribution
        })
        
        df_risk = df_risk[df_risk["Capital Weight"] > 0.001].sort_values(by="Capital Weight", ascending=False)
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            st.markdown("### Capital vs. Risk")
            fig_risk = go.Figure()
            
            fig_risk.add_trace(go.Bar(
                x=df_risk["Ticker"], 
                y=df_risk["Capital Weight"],
                name="Capital Allocation",
                marker_color='blue'
            ))
            
            fig_risk.add_trace(go.Bar(
                x=df_risk["Ticker"], 
                y=df_risk["Risk Contribution"],
                name="Risk Contribution",
                marker_color='orange'
            ))
            
            fig_risk.update_layout(
                barmode='group', 
                yaxis_title="Proportion (0-1)"
            )
            st.plotly_chart(fig_risk, width="stretch")
            
        with col_risk2:
            st.markdown("### Risk Concentration")
            fig_risk_pie = go.Figure(data=[go.Pie(
                labels=df_risk['Ticker'], 
                values=df_risk['Risk Contribution'],
                hole=.4,
                textinfo='label+percent'
            )])
            st.plotly_chart(fig_risk_pie, width="stretch")

elif not run_btn and st.session_state.results is None:
    st.info("Please select tickers and click 'Run Analysis'.")
    st.info(r"""
        **Technical information: built for educational purposes**  
        - Calculations are made assuming there are 252 trading days a year, so crypto and markets with other measurements will be off. Daily close prices are used, not real time data streams.
        - The app doesn't handle exchange rates between different currencies.  
        - Risk metrics assume normal distribution, returns are log-normally distributed  
        - GBM assumes drift rate and volatility is constant. In reality, they are not.
        """)
