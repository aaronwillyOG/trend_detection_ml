import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker: str, period: str = "7d", interval: str = "15m") -> pd.DataFrame:
   
    print(f"Fetching data for {ticker}...")
    
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}. Check your internet or the ticker symbol.")

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    
    df.reset_index(inplace=True)
    
    
    df.columns = [c.lower() for c in df.columns]
    
    
    if 'date' in df.columns:
        df.rename(columns={'date': 'datetime'}, inplace=True)
    
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    print(f"Data loaded successfully: {df.shape[0]} rows.")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    
    if df[required_cols].isnull().any().any():
        print("Warning: Missing values detected. Filling with forward fill.")
        df.fillna(method='ffill', inplace=True)
        
    print("Data validation passed.")
    return True