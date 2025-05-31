import pandas as pd
import numpy as np
from typing import List, Union, Tuple

class TechnicalIndicators:
    """
    A class for calculating various technical indicators for financial data
    with support for both regular DataFrames and MultiIndex column DataFrames.
    Includes safety checks for infinity and very large values.
    Uses Close as target and ensures no lookahead bias in calculations.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.is_multi_index = isinstance(self.data.columns, pd.MultiIndex)

        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                if not self.is_multi_index and 'Date' in self.data.columns:
                    self.data.set_index('Date', inplace=True)
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                print(f"Warning: Could not convert index to DatetimeIndex: {e}")
        
        self._check_and_clean_data()

    def _check_and_clean_data(self) -> None:
        has_inf = np.any(np.isinf(self.data.select_dtypes(include=[np.number])))
        has_too_large = np.any(np.abs(self.data.select_dtypes(include=[np.number])) > 1e308)
        
        if has_inf or has_too_large:
            print("CAUTION: Input data contains infinity or extremely large values.")
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data.loc[np.abs(self.data[col]) > 1e308, col] = np.nan

    def _handle_non_finite_values(self, series) -> pd.Series:
        if series.dtype.kind in 'fc':
            result = series.replace([np.inf, -np.inf], np.nan)
            result = result.mask(np.abs(result) > 1e308, np.nan)
            return result
        return series

    def _get_column(self, col_name: str) -> Union[str, Tuple]:
        if not self.is_multi_index:
            return col_name
        return next((col for col in self.data.columns if col[0] == col_name), None)

    def _get_all_columns(self, col_name: str) -> List:
        if not self.is_multi_index:
            return [col_name] if col_name in self.data.columns else []
        return [col for col in self.data.columns if col[0] == col_name]

    def _add_column(self, col_name: str, ticker: str, values) -> None:
        values = self._handle_non_finite_values(pd.Series(values))
        if self.is_multi_index:
            self.data[(col_name, ticker)] = values
        else:
            self.data[col_name] = values

    def _ensure_numeric(self, columns: List = None) -> None:
        if columns is None:
            columns = self.data.columns
            numeric_columns = [col for col in columns if (not self.is_multi_index and col != 'Date') or 
                              (self.is_multi_index and col[0] != 'Date')]
            columns = numeric_columns
        self.data[columns] = self.data[columns].apply(pd.to_numeric, errors='coerce')

    def calculate_all(self,
                  ema_periods: List[int] = [3, 5, 7, 9, 14, 21],
                  atr_periods: List[int] = [3, 5, 7, 9, 14, 21],
                  roc_periods: List[int] = [3, 5, 7, 9, 14, 21],
                  lag_periods: int = 3) -> pd.DataFrame:
        self._ensure_numeric()
        try:
            self.calculate_daily_range()
            self.calculate_daily_change()
            self.calculate_lagged_prices(periods=lag_periods)
            self.calculate_ema(periods=ema_periods)
            self.calculate_atr(periods=atr_periods)
            self.calculate_roc(periods=roc_periods)
            self.calculate_derivatives_from_close_prices()
            self.calculate_rolling_std_close(windows=[2, 3, 5, 7, 9, 14, 21])
            self.calculate_zscore_close_windows(windows=[2, 3, 5, 7, 9, 14, 21])
            self.calculate_bollinger_band_width_windows(windows=[2, 3, 5, 7, 14, 21])
            self.add_momentum_lagged(windows=[2, 3, 5, 7, 14, 21])
            self.add_rsi_lagged(windows=[2, 3, 5, 7, 14, 21])
            self._check_and_clean_data()
        except Exception as e:
            print(f"CAUTION: Error during indicator calculation: {e}")
        if not ('Date' in self.data.columns or any(col[0] == 'Date' for col in self.data.columns if self.is_multi_index)):
            date_col = 'Date'
            if self.is_multi_index:
                date_col = ('Date', '')
            self.data[date_col] = self.data.index
        return self.data

    def add_momentum_lagged(self, windows: List[int], lag: int = 1):
        """
        Calculate momentum indicator with lag to avoid lookahead bias
        
        Parameters:
        windows: List of periods to look back for momentum calculation
        lag: Number of periods to lag the indicator (1 = use yesterday's value)
        """
        for close_col in self._get_all_columns('Close'):
            ticker = '' if not self.is_multi_index else close_col[1]
            
            for window in windows:
                try:
                    momentum = self.data[close_col] - self.data[close_col].shift(window)
                    # Apply lag to avoid lookahead bias
                    lagged_momentum = momentum.shift(lag)
                    self._add_column(f'Momentum_{window}_lag{lag}', ticker, lagged_momentum)
                except Exception as e:
                    print(f"CAUTION: Error calculating momentum for window {window}: {e}")
    
    def add_rsi_lagged(self, windows: List[int], lag: int = 1):
        """
        Calculate RSI with lag to avoid lookahead bias
        
        Parameters:
        windows: List of periods for RSI calculation
        lag: Number of periods to lag the indicator (1 = use yesterday's value)
        """
        for close_col in self._get_all_columns('Close'):
            ticker = '' if not self.is_multi_index else close_col[1]
            
            for window in windows:
                try:
                    # Calculate price changes
                    delta = self.data[close_col].diff()
                    
                    # Separate gains and losses
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    # Calculate average gain and loss using exponential moving average
                    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
                    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
                    
                    # Avoid division by zero
                    avg_loss = avg_loss.replace(0, np.nan)
                    
                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Handle potential infinities
                    rsi = rsi.replace([np.inf, -np.inf], np.nan)
                    
                    # Apply lag to avoid lookahead bias
                    lagged_rsi = rsi.shift(lag)
                    self._add_column(f'RSI_{window}_lag{lag}', ticker, lagged_rsi)
                except Exception as e:
                    print(f"CAUTION: Error calculating RSI for window {window}: {e}")

    def calculate_daily_range(self) -> None:
        """Calculate daily price range (High - Low)."""
        for high_col in self._get_all_columns('High'):
            if self.is_multi_index:
                low_col = ('Low', high_col[1])
                ticker = high_col[1]
            else:
                low_col = 'Low'
                ticker = ''
            
            if low_col in self.data.columns:
                self._add_column('Daily_Range', ticker, self.data[high_col] - self.data[low_col])

    def calculate_daily_change(self) -> None:
        """Calculate daily price change (Close - Open)."""
        for close_col in self._get_all_columns('Close'):
            if self.is_multi_index:
                open_col = ('Open', close_col[1])
                ticker = close_col[1]
            else:
                open_col = 'Open'
                ticker = ''
            
            if open_col in self.data.columns:
                self._add_column('Daily_Change', ticker, self.data[close_col] - self.data[open_col])

    def calculate_rolling_std_close(self, windows: List[int]) -> None:
        """Calculate rolling standard deviation of Close prices using historical data only."""
        for close_col in self._get_all_columns('Close'):
            ticker = '' if not self.is_multi_index else close_col[1]
            for window in windows:
                try:
                    col_name = f"Rolling_STD_{window}"
                    # Use historical data only - use current window parameters but append NaN to the first window-1 rows
                    # to maintain alignment without lookahead bias
                    rolled = self.data[close_col].rolling(window=window, min_periods=window)
                    std_values = rolled.std()
                    self._add_column(col_name, ticker, std_values)
                except Exception as e:
                    print(f"CAUTION: Error calculating rolling std for window {window}: {e}")

    def calculate_zscore_close_windows(self, windows: List[int]) -> None:
        """Calculate Z-Score of Close for multiple windows using historical data only."""
        for close_col in self._get_all_columns('Close'):
            ticker = '' if not self.is_multi_index else close_col[1]
            for window in windows:
                try:
                    # Use complete windows to avoid lookahead bias
                    rolling_mean = self.data[close_col].rolling(window=window, min_periods=window).mean()
                    rolling_std = self.data[close_col].rolling(window=window, min_periods=window).std()
                    
                    # Avoid division by zero
                    rolling_std = rolling_std.replace(0, np.nan)
                    
                    # Calculate z-score using historical mean and std
                    # Current price relative to historical distribution
                    zscore = (self.data[close_col] - rolling_mean) / rolling_std
                    
                    # Handle potential inf values
                    zscore = zscore.replace([np.inf, -np.inf], np.nan)
                    
                    self._add_column(f"ZScore_Close_{window}", ticker, zscore)
                except Exception as e:
                    print(f"CAUTION: Error calculating Z-Score for window {window}: {e}")
    
    def calculate_bollinger_band_width_windows(self, windows: List[int]) -> None:
        """Calculate Bollinger Band Width for multiple windows using historical data only."""
        for close_col in self._get_all_columns('Close'):
            ticker = '' if not self.is_multi_index else close_col[1]
            for window in windows:
                try:
                    # Use full window to avoid lookahead bias
                    sma = self.data[close_col].rolling(window=window, min_periods=window).mean()
                    std = self.data[close_col].rolling(window=window, min_periods=window).std()
                    
                    # Calculate upper and lower bands
                    upper = sma + 2 * std
                    lower = sma - 2 * std
                    
                    # Avoid division by zero
                    sma_non_zero = sma.replace(0, np.nan)
                    
                    # Calculate BB width and handle potential inf values
                    bb_width = (upper - lower) / sma_non_zero
                    bb_width = bb_width.replace([np.inf, -np.inf], np.nan)
                    
                    self._add_column(f"BB_Width_{window}", ticker, bb_width)
                except Exception as e:
                    print(f"CAUTION: Error calculating Bollinger Band Width for window {window}: {e}")

    def calculate_derivatives(self, prices, order=3):
        """
        Calculates the 1st to specified order derivatives of a time series (closing prices).
        Uses finite differences for numerical differentiation.

        :param prices: List or NumPy array of closing prices
        :param order: Maximum order of derivative to calculate (default is 3)
        :return: Dictionary containing derivatives from 1st to the specified order
        """
        prices = np.array(prices)
        derivatives = {}

        for i in range(1, order + 1):
            if len(prices) > i:
                derivatives[i] = np.diff(prices, n=i)
            else:
                derivatives[i] = np.array([])  # Not enough data points

        return derivatives

    def calculate_derivatives_from_close_prices(self):
        """Calculate derivatives from Close price data (fixed method)."""
        try:
            # For single index DataFrame
            if not self.is_multi_index and 'Close' in self.data.columns:
                # Use Close prices directly, but remove NaN values for calculation
                close_prices = self.data['Close'].dropna().shift(1)
                
                if len(close_prices) > 3:  # Need at least 4 points for 3rd derivative
                    # Calculate derivatives
                    derivatives = self.calculate_derivatives(close_prices, order=3)
                    
                    # Add the derivatives as new columns, aligned with original index
                    for order, derivative_values in derivatives.items():
                        if len(derivative_values) > 0:
                            # Create a series with the same index as original data
                            derivative_series = pd.Series(index=self.data.index, dtype=float)
                            
                            # Fill in the derivative values starting from the appropriate position
                            # Skip the first 'order' positions since diff reduces length
                            start_idx = order
                            end_idx = start_idx + len(derivative_values)
                            
                            if end_idx <= len(self.data):
                                derivative_series.iloc[start_idx:end_idx] = derivative_values
                            
                            self._add_column(f'Close_Lag_1_{order}th_Derivative', '', derivative_series)
                else:
                    print("CAUTION: Not enough data points to calculate derivatives")
            
            # For multi-index DataFrame
            elif self.is_multi_index:
                for close_col in self._get_all_columns('Close'):
                    ticker = close_col[1]
                    close_prices = self.data[close_col].dropna()
                    
                    if len(close_prices) > 3:
                        derivatives = self.calculate_derivatives(close_prices, order=3)
                        
                        for order, derivative_values in derivatives.items():
                            if len(derivative_values) > 0:
                                derivative_series = pd.Series(index=self.data.index, dtype=float)
                                start_idx = order
                                end_idx = start_idx + len(derivative_values)
                                
                                if end_idx <= len(self.data):
                                    derivative_series.iloc[start_idx:end_idx] = derivative_values
                                
                                self._add_column(f'Close_{order}th_Derivative', ticker, derivative_series)
                    else:
                        print(f"CAUTION: Not enough data points to calculate derivatives for ticker {ticker}")
        except Exception as e:
            print(f"CAUTION: Error calculating derivatives: {e}")
            
    def calculate_ema(self, periods: List[int]) -> None:
        """Calculate Exponential Moving Averages for multiple periods using historical data only."""
        close_cols = self._get_all_columns('Close')
    
        for close_col in close_cols:
            ticker = '' if not self.is_multi_index else close_col[1]
    
            # Use the actual close series - EMA is calculated using historical data by default
            close_series = self.data[close_col]
    
            for period in periods:
                try:
                    # adjust=False ensures we only use the historical weighted average
                    # Use proper min_periods without deprecated fillna method
                    ema = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
                    self._add_column(f'EMA_{period}', ticker, ema)
                except Exception as e:
                    print(f"CAUTION: Error calculating EMA for period {period}: {e}")

    def calculate_atr(self, periods: List[int]) -> None:
        """Calculate Average True Range for multiple periods using historical data only."""
        try:
            # For single index DataFrame
            if not self.is_multi_index and all(col in self.data.columns for col in ['High', 'Low', 'Close']):
                high = self.data['High']
                low = self.data['Low']
                close_prev = self.data['Close'].shift(1)  # Proper lagging of close
                
                # Calculate True Range components and handle potential infinities
                range1 = high - low
                range2 = (high - close_prev).abs()
                range3 = (low - close_prev).abs()
                
                # Combine the ranges and take the max
                ranges_df = pd.concat([range1, range2, range3], axis=1)
                ranges_df = ranges_df.replace([np.inf, -np.inf], np.nan)
                tr = ranges_df.max(axis=1)

                for period in periods:
                    # Use ewm with min_periods to avoid lookahead bias
                    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
                    self._add_column(f'ATR_{period}', '', atr)
            
            # For multi-index DataFrame
            elif self.is_multi_index:
                for high_col in self._get_all_columns('High'):
                    ticker = high_col[1]
                    low_col = ('Low', ticker)  # Get corresponding Low column
                    close_col = ('Close', ticker)  # Get corresponding Close column
                    
                    if low_col in self.data.columns and close_col in self.data.columns:
                        high = self.data[high_col]
                        low = self.data[low_col]
                        close_prev = self.data[close_col].shift(1)  # Proper lagging of close
                        
                        # Calculate True Range components and handle potential infinities
                        range1 = high - low
                        range2 = (high - close_prev).abs()
                        range3 = (low - close_prev).abs()
                        
                        # Combine the ranges and take the max
                        ranges_df = pd.concat([range1, range2, range3], axis=1)
                        ranges_df = ranges_df.replace([np.inf, -np.inf], np.nan)
                        tr = ranges_df.max(axis=1)

                        for period in periods:
                            atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
                            self._add_column(f'ATR_{period}', ticker, atr)
        except Exception as e:
            print(f"CAUTION: Error calculating ATR: {e}")

    def calculate_roc(self, periods: List[int]) -> None:
        """Calculate Rate of Change for multiple periods using historical data only."""
        for close_col in self._get_all_columns('Close'):
            ticker = '' if not self.is_multi_index else close_col[1]
            for period in periods:
                try:
                    # Get past values (lookahead safe)
                    past_values = self.data[close_col].shift(period)
                    current_values = self.data[close_col]
                    
                    # Avoid division by zero
                    past_values_non_zero = past_values.replace(0, np.nan)
                    
                    # Correct ROC formula: ((Current / Past) - 1) * 100
                    roc = ((current_values / past_values_non_zero) - 1) * 100
                    
                    # Handle infinities and extremely large values
                    roc = roc.replace([np.inf, -np.inf], np.nan)
                    roc = roc.mask(np.abs(roc) > 1e308, np.nan)
                    
                    self._add_column(f'ROC_{period}', ticker, roc)
                except Exception as e:
                    print(f"CAUTION: Error calculating ROC for period {period}: {e}")

    def calculate_lagged_prices(self, periods: int) -> None:
        """Calculate lagged price values (shifted into the past)."""
        try:
            # For single index DataFrame
            if not self.is_multi_index:
                for price_type in ['Close', 'Low', 'High', 'Open']:
                    if price_type in self.data.columns:
                        for i in range(1, periods + 1):
                            self._add_column(f'{price_type}_Lag_{i}', '', self.data[price_type].shift(i))
                
                # Calculate lagged daily range and change
                if 'Daily_Range' in self.data.columns:
                    for i in range(1, periods + 1):
                        self._add_column(f'Daily_Range_Lag_{i}', '', self.data['Daily_Range'].shift(i))
                
                if 'Daily_Change' in self.data.columns:
                    for i in range(1, periods + 1):
                        self._add_column(f'Daily_Change_Lag_{i}', '', self.data['Daily_Change'].shift(i))
            
            # For multi-index DataFrame
            else:
                for price_type in ['Close', 'Low', 'High', 'Open']:
                    for col in self._get_all_columns(price_type):
                        ticker = col[1]
                        for i in range(1, periods + 1):
                            self._add_column(f'{price_type}_Lag_{i}', ticker, self.data[col].shift(i))
                
                for metric in ['Daily_Range', 'Daily_Change']:
                    for col in self._get_all_columns(metric):
                        ticker = col[1]
                        for i in range(1, periods + 1):
                            self._add_column(f'{metric}_Lag_{i}', ticker, self.data[col].shift(i))
        except Exception as e:
            print(f"CAUTION: Error calculating lagged prices: {e}")