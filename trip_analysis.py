import pandas as pd

def analyze_customer_supplier_trips(filepath):
    # Load the dataset
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Group by customer and supplier, then count trips
    print("Analyzing trip data...")
    trip_counts = df.groupby(['customerNameCode', 'supplierNameCode']).size().reset_index(name='total_trips')
    
    # Sort by customer name and trip count (descending)
    trip_counts = trip_counts.sort_values(by=['customerNameCode', 'total_trips'], ascending=[True, False])
    
    return trip_counts

def main():
    # File path
    filepath = r"c:\Users\mjarj\Desktop\Arjun\TECH VOID\log 2.0\Transportation and Logistics Tracking Dataset.csv"
    
    # Analyze the data
    result = analyze_customer_supplier_trips(filepath)
    
    # Print results in the requested format
    print("\nCustomer – Supplier – Total trips")
    print("-" * 60)
    
    for _, row in result.iterrows():
        print(f"{row['customerNameCode']} – {row['supplierNameCode']} – {row['total_trips']}")
    
    # Save to CSV for reference
    output_file = "customer_supplier_trips.csv"
    result.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
