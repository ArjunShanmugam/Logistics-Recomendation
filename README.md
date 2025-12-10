# 🚚 Logistics Supplier Recommendation System

A machine learning-powered system that recommends the best transportation suppliers for customers based on historical on-time delivery performance, powered by a RandomForestClassifier.

![Dashboard Screenshot](https://img.icons8.com/color/480/delivery--v1.png)

## ✨ Features

- **Smart Supplier Recommendations**: Get top supplier suggestions based on ML predictions
- **Performance Metrics**: View predicted on-time rates, historical performance, and trip volumes
- **Modern Web Interface**: Clean, responsive UI with glassmorphism design
- **Data Analysis**: Built-in tools for analyzing customer-supplier trip data
- **Model Persistence**: Saves trained models for faster subsequent use

## 🛠️ Prerequisites

- Python 3.8+
- pip (Python package manager)
- Modern web browser

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd logistics-supplier-recommendation
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure

```
.
├── app.py                # Streamlit web application
├── logistics_ml.py       # Core ML model and recommendation logic
├── trip_analysis.py      # Script for analyzing customer-supplier trips
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── data/
    └── Transportation and Logistics Tracking Dataset.csv  # Your dataset
```

## 🏃‍♂️ Running the Application

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface** in your browser at `http://localhost:8501`

3. **Select a customer** from the dropdown and click "Recommend suppliers"

## 📊 Additional Tools

### Trip Analysis
To generate a report of all customer-supplier trip counts:
```bash
python trip_analysis.py
```

This will create a `customer_supplier_trips.csv` file with the complete trip data.

## 🤖 ML Model Details

The system uses a RandomForestClassifier with the following features:
- Customer name code
- Supplier name code
- Origin location code
- Destination location code
- Vehicle type
- Transportation distance (KM)

## 📝 Notes

- The first run will train and save the ML model (may take a few minutes)
- Subsequent runs will load the pre-trained model for faster performance
- All data processing is done in-memory - ensure you have sufficient RAM for large datasets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Icons by [Icons8](https://icons8.com/)
- Data visualization powered by [Plotly](https://plotly.com/)
