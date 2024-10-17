import matplotlib.pyplot as plt

def drawStockSeries(df, target_stock):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.title(f'Stock Closing Price Series for {target_stock}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

