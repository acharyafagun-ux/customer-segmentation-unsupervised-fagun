import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    return df

def clean_data(df):
    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove returns and invalid prices
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df


def create_features(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'StockCode': 'nunique'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary', 'UniqueProducts']

    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']

    return rfm


def transform_and_scale(rfm):
    # Log transform
    rfm_log = np.log1p(rfm)

    # Scale
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    return rfm_scaled, rfm_log