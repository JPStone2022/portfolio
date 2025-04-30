import pandas as pd
import numpy as np # Often used alongside Pandas

def analyze_sales_data(data_dict):
    """
    Analyzes sample sales data using Pandas DataFrames.

    Args:
        data_dict (dict): A dictionary representing sales data, suitable
                          for creating a Pandas DataFrame.
                          Expected keys: 'Product', 'Category', 'Price', 'QuantitySold'.

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: The processed DataFrame.
               - pd.DataFrame: Sales summary grouped by category.
               Returns (None, None) if input is invalid.
    """
    try:
        # 1. Create DataFrame
        # In a real scenario, you'd use pd.read_csv('path/to/file.csv'),
        # pd.read_excel(...), pd.read_sql(...), etc.
        df = pd.DataFrame(data_dict)
        print("--- Original Sales Data ---")
        print(df.head()) # Display first 5 rows

        # 2. Initial Data Inspection
        print("\n--- Data Info ---")
        df.info() # Get column types and non-null counts

        print("\n--- Descriptive Statistics (Numerical Columns) ---")
        # .describe() provides summary stats for numerical columns
        print(df.describe())

        # 3. Data Cleaning (Example: Handle potential missing values)
        # Check for missing values
        print("\n--- Missing Values Before Cleaning ---")
        print(df.isnull().sum())

        # Example: Fill missing QuantitySold with the median quantity
        if 'QuantitySold' in df.columns:
            median_quantity = df['QuantitySold'].median()
            df['QuantitySold'].fillna(median_quantity, inplace=True)
            print(f"\nFilled missing 'QuantitySold' with median: {median_quantity}")
            print("Missing Values After Cleaning:")
            print(df.isnull().sum())
        # Note: inplace=True modifies the DataFrame directly

        # Convert Price to numeric if it's not already (handling potential errors)
        if 'Price' in df.columns:
             df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
             # Drop rows where Price could not be converted (became NaN)
             df.dropna(subset=['Price'], inplace=True)


        # 4. Data Selection & Filtering
        print("\n--- Data Selection Examples ---")
        # Select products in the 'Electronics' category
        electronics = df[df['Category'] == 'Electronics']
        print("Electronics Products:\n", electronics)

        # Select products where QuantitySold > 15
        high_quantity = df[df['QuantitySold'] > 15]
        print("\nProducts with Quantity Sold > 15:\n", high_quantity)

        # Select specific columns ('Product', 'Price') for electronics
        electronics_prices = df.loc[df['Category'] == 'Electronics', ['Product', 'Price']]
        print("\nElectronics Product Prices:\n", electronics_prices)

        # 5. Data Manipulation - Creating New Columns
        print("\n--- Data Manipulation ---")
        # Calculate Total Sale for each transaction
        # This uses vectorized operations (element-wise multiplication)
        df['TotalSale'] = df['Price'] * df['QuantitySold']
        print("DataFrame with 'TotalSale' column added:")
        print(df.head())

        # 6. Grouping and Aggregation
        print("\n--- Grouping & Aggregation ---")
        # Calculate total sales amount and average quantity sold per category
        category_summary = df.groupby('Category').agg(
            TotalRevenue=('TotalSale', 'sum'),
            AverageQuantity=('QuantitySold', 'mean'),
            NumberOfProducts=('Product', 'count') # Count products per category
        ).reset_index() # Reset index to make 'Category' a column again

        # Format the summary nicely
        category_summary['AverageQuantity'] = category_summary['AverageQuantity'].round(1)
        category_summary = category_summary.sort_values(by='TotalRevenue', ascending=False)

        print("Sales Summary by Category:")
        print(category_summary)

        return df, category_summary

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None, None

# --- Example Usage ---
if __name__ == "__main__":
    # Sample data (replace with reading from CSV/DB in a real case)
    sales_data = {
        'Product': ['Laptop', 'Keyboard', 'Mouse', 'Monitor', 'Webcam', 'Desk Chair', 'Notebook', 'Pen', 'Laptop', 'Mouse'],
        'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Furniture', 'Stationery', 'Stationery', 'Electronics', 'Electronics'],
        'Price': [1200, 75, 25, 300, 50, 150, 5, 2, 1150, 28],
        'QuantitySold': [10, 25, 30, 8, 15, 5, 50, 100, 12, np.nan] # Include a missing value
    }

    processed_df, summary_df = analyze_sales_data(sales_data)

    if processed_df is not None and summary_df is not None:
        print("\n--- Analysis Completed Successfully ---")
        # You could further process or save these DataFrames here
        # e.g., summary_df.to_csv('category_summary.csv', index=False)
    else:
        print("\n--- Analysis Failed ---")

