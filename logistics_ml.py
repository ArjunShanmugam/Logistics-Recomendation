import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
# Plotting disabled

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Create target variable
    df['is_ontime'] = df['ontime'].apply(lambda x: 1 if x == 'G' else (0 if pd.isna(x) and df.loc[df['ontime'].isna()].index[0] in df[df['delay'] == 'R'].index else np.nan))
    
    # Drop rows where both ontime and delay are missing
    df = df[~((df['ontime'].isna()) & (df['delay'].isna()))]
    
    # Select required columns
    features = [
        'customerNameCode',
        'supplierNameCode',
        'OriginLocation_Code',
        'DestinationLocation_Code',
        'vehicleType',
        'TRANSPORTATION_DISTANCE_IN_KM',
        'is_ontime'
    ]
    
    df = df[features].copy()
    
    # Convert distance to numeric, handle any non-numeric values
    df['TRANSPORTATION_DISTANCE_IN_KM'] = pd.to_numeric(df['TRANSPORTATION_DISTANCE_IN_KM'], errors='coerce')
    
    # Drop rows with missing values in features
    df = df.dropna(subset=features[:-1] + ['is_ontime'])
    
    return df

def train_model(df):
    # Prepare features and target
    categorical_cols = ['customerNameCode', 'supplierNameCode', 'OriginLocation_Code', 'DestinationLocation_Code', 'vehicleType']
    numeric_cols = ['TRANSPORTATION_DISTANCE_IN_KM']
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = encoder.fit_transform(df[categorical_cols])
    X_num = df[numeric_cols].values
    
    # Combine features
    X = np.hstack([X_cat, X_num])
    y = df['is_ontime'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Calculate feature importances (without plotting)
    feature_importances = pd.Series(model.feature_importances_, 
                                  index=list(encoder.get_feature_names_out(categorical_cols)) + numeric_cols)
    print("\nTop 5 most important features:")
    print(feature_importances.nlargest(5).to_string())
    
    return model, encoder, df

def recommend_best_supplier(customer_name, df_raw, model, encoder):
    # Filter data for the specified customer
    customer_data = df_raw[df_raw['customerNameCode'].str.lower() == customer_name.lower()]
    
    if customer_data.empty:
        return f"No data found for customer: {customer_name}"
    
    # Get unique suppliers for this customer
    suppliers = customer_data['supplierNameCode'].unique()
    
    results = []
    
    for supplier in suppliers:
        # Get all trips for this customer-supplier pair
        mask = (df_raw['customerNameCode'].str.lower() == customer_name.lower()) & \
               (df_raw['supplierNameCode'].str.lower() == supplier.lower())
        
        supplier_data = df_raw[mask].copy()
        
        # Calculate historical metrics
        total_trips = len(supplier_data)
        historical_ontime_rate = supplier_data['is_ontime'].mean()
        
        # Prepare features for prediction
        if not supplier_data.empty:
            # Take the most recent trip's features for prediction
            sample = supplier_data.iloc[0:1].copy()
            
            # One-hot encode the features
            categorical_cols = ['customerNameCode', 'supplierNameCode', 'OriginLocation_Code', 
                              'DestinationLocation_Code', 'vehicleType']
            X_cat = encoder.transform(sample[categorical_cols])
            X_num = sample[['TRANSPORTATION_DISTANCE_IN_KM']].values
            X = np.hstack([X_cat, X_num])
            
            # Get predicted probability of on-time delivery
            proba = model.predict_proba(X)[0][1]  # Probability of class 1 (on-time)
            
            results.append({
                'supplier': supplier,
                'predicted_ontime': proba,
                'historical_ontime': historical_ontime_rate,
                'trip_count': total_trips
            })
    
    # Sort by predicted on-time probability (descending)
    results = sorted(results, key=lambda x: x['predicted_ontime'], reverse=True)
    
    # Return top 3 suppliers
    return results[:3]

def get_total_trips(df_raw, customer_name, supplier_name):
    return len(df_raw[(df_raw['customerNameCode'].str.lower() == customer_name.lower()) & 
                     (df_raw['supplierNameCode'].str.lower() == supplier_name.lower())])

def main():
    # File path
    filepath = r"c:\Users\mjarj\Desktop\Arjun\TECH VOID\log 2.0\Transportation and Logistics Tracking Dataset.csv"
    model_file = 'logistics_model.joblib'
    encoder_file = 'logistics_encoder.joblib'
    
    # Load raw data for total trips count
    print("Loading raw data for trip counts...")
    df_raw = pd.read_csv(filepath)
    
    # Check if model exists
    if os.path.exists(model_file) and os.path.exists(encoder_file):
        print("Loading pre-trained model...")
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        # Load just enough data for the model to work
        df = load_and_preprocess_data(filepath)
    else:
        # Load and preprocess data for model
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data(filepath)
        
        # Train new model
        print("Training new model (this may take a minute)...")
        model, encoder, _ = train_model(df)
        
        # Save the model and encoder
        joblib.dump(model, model_file)
        joblib.dump(encoder, encoder_file)
        print("Model trained and saved for future use.")
    
    # Interactive mode for customer input
    print("\n" + "="*60)
    print("Supplier Recommendation System")
    print("="*60)
    
    while True:
        print("\nEnter a customer name (or 'exit' to quit):")
        customer_name = input("> ").strip()
        
        if customer_name.lower() == 'exit':
            print("Exiting...")
            break
            
        if not customer_name:
            print("Please enter a valid customer name.")
            continue
            
        print(f"\nFinding best suppliers for: {customer_name}")
        recommendations = recommend_best_supplier(customer_name, df, model, encoder)
        
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            if not recommendations:
                print(f"No supplier data found for customer: {customer_name}")
                continue
                
            print(f"\nBest suppliers for {customer_name}:")
            print("-" * 100)
            print(f"{'#':<3} {'Supplier':<40} {'Predicted On-time':<20} {'Historical Rate':<20} {'Trips (All)'}")
            print("-" * 100)
            
            for i, rec in enumerate(recommendations, 1):
                total_trips = get_total_trips(df_raw, customer_name, rec['supplier'])
                print(f"{i:<3} {rec['supplier']:<40} {rec['predicted_ontime']:.2f}{' ':<15} {rec['historical_ontime']:.2f}{' ':<15} {total_trips}")
            
            print("\nNote: Suppliers are ranked by predicted on-time delivery probability.")
            print("      Historical rate shows actual on-time delivery percentage.")
            print("      Trips (All) shows the total number of trips including all data.")
        
        print("\n" + "="*100)

if __name__ == "__main__":
    main()
