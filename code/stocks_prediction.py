import yfinance as yf
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt


# Load stocks list from CSV into a DataFrame
def get_stock_symbols(stocks_list_filename):
    data_directory = "../data"
    stocks_list_filename = stocks_list_filename
    stocks_list_path = os.path.join(data_directory, stocks_list_filename)
    # return stock_symbols_df
    return pd.read_csv(stocks_list_path)


stock_symbols_df = get_stock_symbols("TickerSymbols.csv")
print(stock_symbols_df.head(5), "\n")
print(stock_symbols_df.tail(5))

# Take a random sample of 50 elements from stock_symbols_df for code testing
# You can adjust the number of samples and the random state as needed
stock_symbols_df = stock_symbols_df.sample(n=50, random_state=42)
print(stock_symbols_df.head(5), "\n")
print(stock_symbols_df.tail(5))


def get_stock_data(df):
    # Create variables to count symbols evaluated and symbols not found
    symbols_evaluated = 0
    symbols_not_found = 0

    # Create a list to store stock dataframes
    stock_dfs = []

    # Fetch historical stock prices for the last 3 months using yfinance
    for index, row in df.iterrows():
        symbol = row['symbol']
        name = row['name']
        try:
            stock = yf.Ticker(symbol)
            period = "3mo"
            history = stock.history(period=period)  # Fetching data for the last 90 days or 3 months
            if not history.empty:  # Check if data is available
                history.reset_index(inplace=True)
                history['Symbol'] = symbol  # Add Symbol column
                history['Name'] = name  # Add Name column
                history['Date'] = history['Date'].dt.date  # Extract only the date portion
                stock_dfs.append(
                    history[['Date', 'Symbol', 'Name', 'Close']])  # Keep only Date, Symbol, Name and Close columns
                symbols_evaluated += 1
            else:
                # print(f"No data found for symbol {symbol}.")
                symbols_not_found += 1
        except yf.YFinanceError as e:
            print(f"Failed to fetch data for symbol {symbol}: {e}")
            symbols_not_found += 1

    print("\nSymbols Evaluated:", symbols_evaluated)
    print("Symbols Not Found:", symbols_not_found)

    # Concatenate all stock dataframes into a single dataframe
    print(len(stock_dfs))
    # return stock_data
    return pd.concat(stock_dfs, ignore_index=True)


stock_data = get_stock_data(stock_symbols_df)
print(len(stock_data))
print(stock_data.head(5), "\n")
print(stock_data.tail(5))


# Define a custom aggregation function
def custom_agg_dates_close_return(x):
    min_date = x['Date'].idxmin()
    max_date = x['Date'].idxmax()
    return pd.Series({
        'min_date': x.loc[min_date, 'Date'],
        'max_date': x.loc[max_date, 'Date'],
        'Close_min_date': x.loc[min_date, 'Close'],
        'Close_max_date': x.loc[max_date, 'Close'],
        'Return (%)': ((x.loc[max_date, 'Close'] - x.loc[min_date, 'Close']) / x.loc[min_date, 'Close']) * 100
    })


# Group by Symbol and perform custom aggregation
stocks_growth = stock_data.groupby('Symbol').apply(custom_agg_dates_close_return, include_groups=False)

# Reset index to obtain a DataFrame
stocks_growth_df = stocks_growth.reset_index()

# Sort the DataFrame by "Return (%)" column in descending order
stocks_growth_df = stocks_growth_df.sort_values(by='Return (%)', ascending=False)

# Display the resulting DataFrame
print(stocks_growth_df.head(7), "\n")
print(stocks_growth_df.tail(7))


# Save stocks growth findings to a csv file
def save_stocks_growth_to_csv(x):
    # Define output directory and filename
    csv_directory = "../output/csv"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stocks_growth_filename = f"stocks_growth_{current_datetime}.csv"
    stocks_growth_path = os.path.join(csv_directory, stocks_growth_filename)

    # Save the DataFrame to a CSV file
    x.to_csv(stocks_growth_path, index=False)

    # Print the filename for confirmation
    print(f"DataFrame saved to {stocks_growth_path}")


save_stocks_growth_to_csv(stocks_growth_df)


def plot_top_n_stocks(x, n=25):
    # Select the top n symbols by return
    top_n_stocks_increasing = x.head(n)
    top_n_stocks_decreasing = x.tail(n)

    # Define output directory and filename
    png_directory = "../output/png"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename_increasing_and_decreasing = f"top_{n}_stocks_increasing_and_decreasing_{current_datetime}.png"
    plot_filename_increasing_and_decreasing_path = os.path.join(png_directory, plot_filename_increasing_and_decreasing)

    # Create the first plot: Top n Stocks Increasing
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.bar(top_n_stocks_increasing['Symbol'], top_n_stocks_increasing['Return (%)'], color='skyblue')
    plt.xlabel('Symbol')
    plt.ylabel('Return (%)')
    plt.title(f'Top {n} Stocks Increasing\n Period: {x["min_date"].min()}  to  {x["max_date"].max()}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Create the second plot: Top n Stocks Decreasing
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.bar(top_n_stocks_decreasing['Symbol'], top_n_stocks_decreasing['Return (%)'], color='salmon')
    plt.xlabel('Symbol')
    plt.ylabel('Return (%)')
    plt.title(f'Top {n} Stocks Decreasing\n Period: {x["min_date"].min()}  to  {x["max_date"].max()}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the 2 subplots
    plt.savefig(plot_filename_increasing_and_decreasing_path)
    # Print the filename for confirmation
    print(f"Plots saved to {plot_filename_increasing_and_decreasing_path}")

    # Show both plots
    plt.show()


# Example usage with default n=25
plot_top_n_stocks(stocks_growth_df)


# Example usage, with n=50
plot_top_n_stocks(stocks_growth_df, 50)
