import yfinance as yf
import pandas as pd

def fetch_tesla_september_2018():
    """
    Fetch Tesla stock data for September 2018
    Saves as Excel file for easy importing into Excel/Google Sheets
    """
    print("Fetching Tesla (TSLA) data for September 2018...\n")
    
    # Fetch Tesla data
    tsla = yf.Ticker("TSLA")
    hist = tsla.history(start="2018-09-01", end="2018-09-30")
    
    # Prepare dataframe
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Adj Close'] = hist['Close']  # Already adjusted in newer yfinance
    df.reset_index(inplace=True)
    
    # Format date
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Calculate daily returns
    df['Daily Return (%)'] = df['Close'].pct_change() * 100  # As percentage
    df['Daily Return (%)'] = df['Daily Return (%)'].round(2)
    
    # Round price columns to 2 decimals
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        df[col] = df[col].round(2)
    
    # Reorder columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily Return (%)']]
    
    print("=" * 80)
    print("SAVING DATA TO FILES")
    print("=" * 80)
    
    # Save to CSV (always works)
    csv_filename = 'tesla_september_2018.csv'
    df.to_csv(csv_filename, index=False)
    print(f"✓ CSV saved: {csv_filename}")
    
    # Try to save to Excel
    try:
        excel_filename = 'tesla_september_2018.xlsx'
        df.to_excel(excel_filename, index=False, sheet_name='Tesla Sep 2018')
        print(f"✓ Excel saved: {excel_filename}")
    except ImportError:
        print("⚠ Excel export requires openpyxl. Install with: pip install openpyxl")
        print("  (CSV file is still available)")
    
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total trading days: {len(df)}")
    print(f"\nKey observations:")
    
    # Find September 7 data
    sep7 = df[df['Date'] == '2018-09-07']
    if not sep7.empty:
        ret = sep7['Daily Return (%)'].values[0]
        vol = sep7['Volume'].values[0]
        print(f"  • September 7, 2018 (Joe Rogan podcast event):")
        print(f"    - Daily return: {ret}%")
        print(f"    - Volume: {vol:,.0f}")
        print(f"    - Volume vs monthly avg: {vol / df['Volume'].mean():.2f}x")
    
    print(f"\n  • Monthly statistics:")
    print(f"    - Average daily return: {df['Daily Return (%)'].mean():.2f}%")
    print(f"    - Volatility (std dev): {df['Daily Return (%)'].std():.2f}%")
    print(f"    - Min return: {df['Daily Return (%)'].min():.2f}%")
    print(f"    - Max return: {df['Daily Return (%)'].max():.2f}%")
    print(f"    - Average volume: {df['Volume'].mean():,.0f}")
    
    print("\n" + "=" * 80)
    print("FILES READY!")
    print("=" * 80)
    print("Open 'tesla_september_2018.xlsx' in Excel or")
    print("Import 'tesla_september_2018.csv' into Google Sheets")
    print("=" * 80)

if __name__ == "__main__":
    try:
        fetch_tesla_september_2018()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
