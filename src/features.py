import pandas as pd
import numpy as np

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    
    
    df = df.copy()

    
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    
    df['rsi'] = calculate_rsi(df['close'])

    df['volatility'] = df['close'].rolling(window=20).std()

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)

    print(f"Features created. Data shape: {df.shape}")
    print(f"   Target Distribution: {df['target'].value_counts(normalize=True).to_dict()}")
    
    return df