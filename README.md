# Rate of Return on Everything Dashboard

An interactive Streamlit dashboard for analyzing the comprehensive "Rate of Return on Everything" dataset, which contains long-term historical returns on various asset classes.

## Features

- **📋 Data Overview**: Explore dataset structure, summary statistics, and data types
- **📈 Time Series Analysis**: Analyze return trends over time with interactive year selection and rolling statistics
- **📊 Asset Comparison**: Compare multiple assets with correlation matrices, distribution plots, and cumulative performance
- **💰 Return Analysis**: Detailed return statistics including Sharpe ratios, risk-return profiles, and distribution analysis
- **🔍 Custom Analysis**: Create custom visualizations and apply filters to explore specific aspects of the data

## Installation

1. **Clone or download** the project files to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data file location**: The dashboard expects the Stata data file at:
   ```
   ./6389799/RORE_QJE_replication_v2/data/rore_public_main.dta
   ```

## Usage

1. **Navigate to the project directory** in your terminal/command prompt

2. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_dashboard.py
   ```

3. **Open your web browser** and go to the URL displayed in the terminal (usually `http://localhost:8501`)

## Data Source

This dashboard works with the "Rate of Return on Everything" dataset, which contains comprehensive historical return data on various asset classes including:
- Stocks
- Bonds
- Real estate
- Commodities
- Art and collectibles
- And many other asset types

## Dashboard Sections

### Data Overview
- First 10 rows of the dataset
- Summary statistics for numerical variables
- Data type information and null value counts

### Time Series Analysis
- Interactive year range selection
- Return trends over time for selected assets
- Rolling statistics (mean and standard deviation)

### Asset Comparison
- Correlation matrix between selected assets
- Box plots for return distribution comparison
- Cumulative performance comparison

### Return Analysis
- Key metrics: mean return, volatility, Sharpe ratio, maximum return
- Return distribution histograms
- Risk-return scatter plots

### Custom Analysis
- Create custom scatter plots between any two variables
- Add trend lines to visualizations
- Apply custom filters to explore specific data subsets

## Customization

The dashboard automatically detects return-related columns by looking for keywords like 'return', 'ret', or 'rate' in column names. You can modify the `return_cols` detection logic in the code if your data uses different naming conventions.

## Troubleshooting

- **Data file not found**: Ensure the Stata file is in the correct location as specified in the code
- **Missing dependencies**: Run `pip install -r requirements.txt` to install all required packages
- **Large dataset issues**: The dashboard uses caching to improve performance with large datasets

## Deployment to Render

This dashboard can be deployed to Render using the included `render.yaml` configuration file.

### Steps to Deploy:

1. **Push your code to a Git repository** (GitHub, GitLab, or Bitbucket)

2. **Connect to Render**:
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" and select "Blueprint"
   - Connect your repository

3. **Render will automatically detect `render.yaml`** and configure the service

4. **Ensure data file is in repository**:
   - The data file should be at: `./6389799/RORE_QJE_replication_v2/data/rore_public_main.dta`
   - Or update the paths in `streamlit_dashboard.py` if using a different location

5. **Deploy**: Render will automatically build and deploy your application

### Manual Deployment (Alternative):

If not using `render.yaml`, create a new Web Service on Render with:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run streamlit_dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- **Environment**: Python 3
- **Plan**: Free (or upgrade for better performance)

## Requirements

- Python 3.8+
- Streamlit 1.28+
- Pandas 2.0+
- NumPy 1.24+
- Plotly 5.15+
- Pyreadstat 1.1.0+ (for reading Stata files)

## License

This project is provided as-is for educational and research purposes.
