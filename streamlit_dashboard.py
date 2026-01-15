import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try to import Stata file readers - prefer pandas as it's more reliable
STATA_READER = None
try:
    # Check if pandas can read Stata files (pandas 1.0+)
    if hasattr(pd, 'read_stata'):
        STATA_READER = 'pandas'
    else:
        raise AttributeError("pandas.read_stata not available")
except (AttributeError, Exception):
    try:
        import pyreadstat
        STATA_READER = 'pyreadstat'
    except ImportError:
        STATA_READER = None

# Page configuration
st.set_page_config(
    page_title="Rate of Return on Everything Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        display: none;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_stata_data(df):
    """Preprocess Stata data to handle common issues"""
    df_processed = df.copy()
    
    # Handle common Stata data type issues
    for col in df_processed.columns:
        # Check if column contains mixed types or problematic values
        if df_processed[col].dtype == 'object':
            # Try to convert to numeric if possible
            try:
                numeric_col = pd.to_numeric(df_processed[col], errors='coerce')
                if not numeric_col.isna().all():
                    df_processed[col] = numeric_col
            except:
                pass
        
        # Handle timestamp columns that might be read incorrectly
        if 'year' in col.lower() or 'date' in col.lower():
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                # Extract year from timestamp
                df_processed[f'{col}_numeric'] = df_processed[col].dt.year
    
    return df_processed

def calculate_rolling_geometric_mean(returns, window):
    """Calculate rolling geometric mean of returns"""
    # Convert returns to (1 + returns) format
    returns_plus_one = 1 + returns
    
    # Calculate rolling geometric mean using product and nth root with raw=True
    rolling_geometric_mean = returns_plus_one.rolling(window=window, min_periods=window).apply(
        lambda x: np.prod(x) ** (1/window) - 1 if len(x) == window else np.nan, raw=True
    ).shift(-window)
    
    return rolling_geometric_mean

def calculate_rolling_total_return(returns, window):
    """Calculate total percentage return over rolling periods"""
    # Convert returns to (1 + returns) format
    returns_plus_one = 1 + returns
    
    # Calculate rolling product and shift forward, then subtract 1 with raw=True
    rolling_total = returns_plus_one.rolling(window=window, min_periods=window).apply(
        lambda x: np.prod(x) - 1 if len(x) == window else np.nan, raw=True
    ).shift(-window)
    
    return rolling_total

def calculate_rolling_statistics(returns, window):
    """Calculate rolling statistics for returns"""
    rolling_mean = calculate_rolling_geometric_mean(returns, window)
    rolling_total_return = calculate_rolling_total_return(returns, window)
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    rolling_upper = rolling_mean + rolling_std
    rolling_lower = rolling_mean - rolling_std
    
    # Calculate percentage of rolling periods that have negative total returns
    # This matches the notebook approach: (data < 0).sum().sum()/data.count()
    valid_rolling_returns = rolling_total_return.dropna()
    if len(valid_rolling_returns) > 0:
        negative_periods_pct = (valid_rolling_returns < 0).sum() / len(valid_rolling_returns) * 100
    else:
        negative_periods_pct = 0
    
    # Create a series with the same length as input, filled with the calculated percentage
    rolling_negative_pct = pd.Series([negative_periods_pct] * len(returns), index=returns.index)
    
    return {
        'mean': rolling_mean,
        'total_return': rolling_total_return,
        'std': rolling_std,
        'upper': rolling_upper,
        'lower': rolling_lower,
        'negative_pct': rolling_negative_pct
    }

@st.cache_data
def load_data():
    """Load the Rate of Return on Everything dataset"""
    import os
    
    # First, try to load from Parquet (preferred - faster and no encoding issues)
    parquet_paths = [
        "./6389799/RORE_QJE_replication_v2/data/rore_public_main.parquet",
        "6389799/RORE_QJE_replication_v2/data/rore_public_main.parquet",
        "../6389799/RORE_QJE_replication_v2/data/rore_public_main.parquet",
        "./data/rore_public_main.parquet"
    ]
    
    for path in parquet_paths:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path, engine='pyarrow')
                return df
            except Exception as e:
                # Only show warning if it's the last path, to avoid cluttering
                if path == parquet_paths[-1]:
                    st.warning(f"Error reading Parquet file at {path}: {str(e)}")
                continue
    
    # Fallback to Stata file if Parquet not available
    if STATA_READER is None:
        st.error("No Stata file reader available. Please install pyreadstat or pandas with Stata support.")
        return None
    
    # Try multiple possible paths for the Stata file
    possible_paths = [
        "./6389799/RORE_QJE_replication_v2/data/rore_public_main.dta",
        "6389799/RORE_QJE_replication_v2/data/rore_public_main.dta",
        "../6389799/RORE_QJE_replication_v2/data/rore_public_main.dta",
        "./data/rore_public_main.dta"
    ]
    
    for path in possible_paths:
        try:
            # Check if file exists first
            if not os.path.exists(path):
                continue
            
            # Read the file - prefer pandas as it handles binary files more reliably
            if STATA_READER == 'pandas':
                # Use pandas with binary file handle - most reliable method
                # pandas.read_stata handles encoding internally
                with open(path, 'rb') as f:
                    df = pd.read_stata(f)
            elif STATA_READER == 'pyreadstat':
                # pyreadstat with explicit encoding to avoid UTF-8 issues
                # The encoding parameter affects how metadata (variable labels) are read
                try:
                    # Try with latin1 encoding first (common for Stata files)
                    df, meta = pyreadstat.read_dta(path, encoding='latin1')
                except (UnicodeDecodeError, ValueError) as e:
                    # If encoding error, try with iso-8859-1 (similar to latin1)
                    try:
                        df, meta = pyreadstat.read_dta(path, encoding='iso-8859-1')
                    except Exception:
                        # Last resort: try without encoding (may fail but worth trying)
                        df, meta = pyreadstat.read_dta(path)
            else:
                # Fallback: try pandas if it's available even if not detected
                try:
                    with open(path, 'rb') as f:
                        df = pd.read_stata(f)
                except Exception as e:
                    raise Exception(f"Unable to read Stata file. Error: {str(e)}")
            
            # Preprocess the data to handle common Stata issues
            df = preprocess_stata_data(df)
            
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            # Only show warning if file exists (to avoid cluttering with path errors)
            if os.path.exists(path):
                st.warning(f"Error reading file at {path}: {str(e)}")
            continue
    
    # If none of the paths work, show error with current directory info
    current_dir = os.getcwd()
    st.error("Data file not found. Please ensure the Stata file is in the correct location.")
    st.info(f"Current working directory: {current_dir}")
    st.info("Expected file: rore_public_main.dta")
    st.info("Available paths tried:")
    for path in possible_paths:
        exists = os.path.exists(path) if path else False
        st.info(f"  - {path} {'(EXISTS)' if exists else '(NOT FOUND)'}")
    return None

def main():
    st.markdown('<h1 class="main-header">📈 Rate of Return on Everything Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Preprocess data - handle year column conversion
    if 'year' in df.columns:
        year_col = df['year']
        try:
            if pd.api.types.is_datetime64_any_dtype(year_col):
                # Convert timestamp to year for easier analysis
                df['year_numeric'] = year_col.dt.year
            else:
                # Ensure year is numeric
                df['year_numeric'] = pd.to_numeric(year_col, errors='coerce')
                if df['year_numeric'].isna().any():
                    st.warning("Some year values could not be converted to numeric format")
        except Exception as e:
            st.error(f"Error processing year column: {e}")
            # Create a fallback year column
            df['year_numeric'] = range(len(df))
    else:
        st.warning("No 'year' column found. Creating sequential index for time series analysis.")
        df['year_numeric'] = range(len(df))
    
    # Main content area - KPI Dashboard only
    st.header("🎯 KPI Dashboard - Country Analysis")
    
    # Create two-column layout
    col_left, col_right = st.columns([3, 1])
    
    with col_right:
        st.subheader("🎛️ Controls")
        
        # Country selection
        if 'country' in df.columns:
            countries = sorted(df['country'].unique())
            selected_country = st.selectbox("Select Country:", countries, index=countries.index('AUS') if 'AUS' in countries else 0)
            
            # Filter data for selected country
            country_data = df[df['country'] == selected_country].copy()
            
            if len(country_data) > 0:
                # Find all return columns for the selected country (numeric only)
                potential_cols = [col for col in country_data.columns if any(keyword in col.lower() for keyword in ['tr', 'return', 'ret'])]
                
                # Columns to exclude from return type options
                excluded_cols = ['eq_tr_interp', 'capital_tr']
                
                # Filter to only numeric columns and exclude specified columns
                return_cols = []
                for col in potential_cols:
                    # Skip excluded columns
                    if col in excluded_cols:
                        continue
                    try:
                        # Try to convert to numeric, if successful, it's a valid return column
                        pd.to_numeric(country_data[col], errors='raise')
                        return_cols.append(col)
                    except (ValueError, TypeError):
                        # Skip non-numeric columns
                        continue
                
                # Return type selection
                st.subheader("📊 Return Type")
                if len(return_cols) >= 2:
                    selected_return = st.selectbox("Primary Return Type:", return_cols, index=0)
                    
                    # Secondary return type (for comparison)
                    secondary_options = [col for col in return_cols if col != selected_return]
                    if secondary_options:
                        secondary_return = st.selectbox("Secondary Return Type:", secondary_options, index=0)
                    else:
                        secondary_return = selected_return
                else:
                    st.warning("Need at least 2 return columns for analysis")
                    st.stop()
                
                # Timeframe selection
                st.subheader("⏰ Timeframe")
                if 'year_numeric' in country_data.columns:
                    years = sorted(country_data['year_numeric'].unique())
                    if len(years) > 1:
                        # Find index for 1951, default to 0 if not found
                        default_start_index = 0
                        if 1951 in years:
                            default_start_index = years.index(1951)
                        start_year = st.selectbox("Start Year:", years, index=default_start_index)
                        end_year = st.selectbox("End Year:", years, index=len(years)-1)
                        
                        # Filter data by timeframe
                        timeframe_data = country_data[
                            (country_data['year_numeric'] >= start_year) & 
                            (country_data['year_numeric'] <= end_year)
                        ].copy()
                    else:
                        timeframe_data = country_data.copy()
                        start_year = end_year = years[0]
                else:
                    timeframe_data = country_data.copy()
                    start_year = end_year = "N/A"
                
                # Rolling period selection
                st.subheader("📊 Rolling Analysis")
                max_rolling_period = len(timeframe_data) if len(timeframe_data) > 0 else 1
                rolling_period = st.slider(
                    "Rolling Period (years):", 
                    min_value=1, 
                    max_value=max_rolling_period, 
                    value=min(5, max_rolling_period),
                    help="Number of years to include in rolling calculations"
                )
                
                # Analysis type selection
                analysis_type = st.radio(
                    "Analysis Type:",
                    ["Annual Returns", "Rolling Returns"],
                    help="Annual: Show individual year returns. Rolling: Show rolling geometric mean returns."
                )
                
                # Toggle options
                st.subheader("🔧 Display Options")
                show_secondary = st.checkbox("Show Secondary Return Type", value=True)
                show_statistical_bands = st.checkbox("Show Statistical Bands", value=True)
                
                # Show what columns we found
                st.info(f"**Primary:** {selected_return}")
                st.info(f"**Secondary:** {secondary_return}")
                st.info(f"**Timeframe:** {start_year} - {end_year}")
                
                if len(timeframe_data) > 0:
                    try:
                        # Ensure columns are numeric before calculating statistics
                        timeframe_data[selected_return] = pd.to_numeric(timeframe_data[selected_return], errors='coerce')
                        timeframe_data[secondary_return] = pd.to_numeric(timeframe_data[secondary_return], errors='coerce')
                        
                        # Sort by year to ensure proper rolling calculations
                        if 'year_numeric' in timeframe_data.columns:
                            timeframe_data = timeframe_data.sort_values('year_numeric').reset_index(drop=True)
                        
                        if analysis_type == "Rolling Returns":
                            # Calculate rolling statistics
                            primary_rolling = calculate_rolling_statistics(timeframe_data[selected_return], rolling_period)
                            secondary_rolling = calculate_rolling_statistics(timeframe_data[secondary_return], rolling_period)
                            
                            # Use rolling statistics for KPIs
                            primary_mean = primary_rolling['mean'].mean()
                            primary_total_return = primary_rolling['total_return'].mean()
                            primary_std = primary_rolling['std'].mean()
                            primary_upper = primary_rolling['upper'].mean()
                            primary_lower = primary_rolling['lower'].mean()
                            primary_pc_negative = primary_rolling['negative_pct'].iloc[0]  # Get the single percentage value
                            
                            secondary_mean = secondary_rolling['mean'].mean()
                            secondary_total_return = secondary_rolling['total_return'].mean()
                            secondary_std = secondary_rolling['std'].mean()
                            secondary_upper = secondary_rolling['upper'].mean()
                            secondary_lower = secondary_rolling['lower'].mean()
                            secondary_pc_negative = secondary_rolling['negative_pct'].iloc[0]  # Get the single percentage value
                            
                            # Calculate number of rolling periods where asset 1 beats asset 2
                            # Compare rolling means (geometric mean returns)
                            primary_rolling_means = primary_rolling['mean'].dropna()
                            secondary_rolling_means = secondary_rolling['mean'].dropna()
                            
                            # Align indices for comparison
                            common_indices = primary_rolling_means.index.intersection(secondary_rolling_means.index)
                            if len(common_indices) > 0:
                                asset1_beats_asset2_count = (primary_rolling_means.loc[common_indices] > secondary_rolling_means.loc[common_indices]).sum()
                                total_comparable_periods = len(common_indices)
                                asset1_beats_asset2_pct = (asset1_beats_asset2_count / total_comparable_periods * 100) if total_comparable_periods > 0 else 0
                            else:
                                asset1_beats_asset2_count = 0
                                total_comparable_periods = 0
                                asset1_beats_asset2_pct = 0
                            
                            # Store rolling data for plotting
                            primary_rolling_data = primary_rolling
                            secondary_rolling_data = secondary_rolling
                        else:
                            # Calculate annual statistics (original logic)
                            primary_mean = timeframe_data[selected_return].mean()
                            primary_std = timeframe_data[selected_return].std()
                            primary_upper = primary_mean + primary_std
                            primary_lower = primary_mean - primary_std
                            primary_pc_negative = (timeframe_data[selected_return] < 0).mean() * 100
                            
                            secondary_mean = timeframe_data[secondary_return].mean()
                            secondary_std = timeframe_data[secondary_return].std()
                            secondary_upper = secondary_mean + secondary_std
                            secondary_lower = secondary_mean - secondary_std
                            secondary_pc_negative = (timeframe_data[secondary_return] < 0).mean() * 100
                            
                            # Calculate number of annual periods where asset 1 beats asset 2
                            asset1_beats_asset2_count = (timeframe_data[selected_return] > timeframe_data[secondary_return]).sum()
                            total_comparable_periods = len(timeframe_data)
                            asset1_beats_asset2_pct = (asset1_beats_asset2_count / total_comparable_periods * 100) if total_comparable_periods > 0 else 0
                            
                            # Set rolling data to None for annual analysis
                            primary_rolling_data = None
                            secondary_rolling_data = None
                        
                    except Exception as e:
                        st.error(f"Error calculating statistics: {e}")
                        st.error(f"Selected return column '{selected_return}' may contain non-numeric data")
                        st.error(f"Secondary return column '{secondary_return}' may contain non-numeric data")
                        st.stop()
                    
                    # Create KPI Widgets using Plotly
                    with col_left:
                        st.subheader("📊 Key Performance Indicators")
                        
                        # Create the KPI grid layout
                        fig_kpi = go.Figure()
                        
                        # Primary Return Type KPIs (Row 0)
                        if analysis_type == "Rolling Returns":
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_mean*100, 2),
                                title={'text': f"Geometric Mean {selected_return}", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 0}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_total_return*100, 2),
                                title={'text': f"Total Return {selected_return} %", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 1}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_upper*100, 2),
                                title={'text': f"{selected_return} +stddev", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 2}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_pc_negative, 2),
                                title={'text': f"{selected_return} Negative Periods %", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 3}
                            ))
                        else:
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_mean*100, 2),
                                title={'text': f"Mean {selected_return}", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 0}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_upper*100, 2),
                                title={'text': f"{selected_return} +stddev", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 1}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_lower*100, 2),
                                title={'text': f"{selected_return} -stddev", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 2}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(primary_pc_negative, 2),
                                title={'text': f"{selected_return} Negative Years %", 'font_color': '#A23B72'},
                                number={"font": {"size": 40}, 'font_color': '#A23B72'},
                                domain={'row': 0, 'column': 3}
                            ))
                        
                        # Secondary Return Type KPIs (Row 1)
                        if analysis_type == "Rolling Returns":
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_mean*100, 2),
                                title={'text': f"Geometric Mean {secondary_return}", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 0}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_total_return*100, 2),
                                title={'text': f"Total Return {secondary_return} %", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 1}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_upper*100, 2),
                                title={'text': f"{secondary_return} +stddev", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 2}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_pc_negative, 2),
                                title={'text': f"{secondary_return} Negative Periods %", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 3}
                            ))
                        else:
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_mean*100, 2),
                                title={'text': f"Mean {secondary_return}", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 0}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_upper*100, 2),
                                title={'text': f"{secondary_return} +stddev", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 1}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_lower*100, 2),
                                title={'text': f"{secondary_return} -stddev", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 2}
                            ))
                            
                            fig_kpi.add_trace(go.Indicator(
                                value=round(secondary_pc_negative, 2),
                                title={'text': f"{secondary_return} Negative Years %", 'font_color': '#2E86AB'},
                                number={"font": {"size": 40}, 'font_color': '#2E86AB'},
                                domain={'row': 1, 'column': 3}
                            ))
                        
                        # Comparison KPI (Row 2) - Asset 1 beats Asset 2
                        period_label = f"{rolling_period}-Year Rolling Periods" if analysis_type == "Rolling Returns" else "Annual Periods"
                        fig_kpi.add_trace(go.Indicator(
                            value=asset1_beats_asset2_count,
                            title={'text': f"{selected_return} Beats {secondary_return} ({period_label})", 'font_color': '#F18F01'},
                            number={"font": {"size": 50}, 'font_color': '#F18F01', "suffix": f" / {total_comparable_periods}"},
                            domain={'row': 2, 'column': 0}
                        ))
                        
                        fig_kpi.add_trace(go.Indicator(
                            value=round(asset1_beats_asset2_pct, 2),
                            title={'text': f"{selected_return} Beats {secondary_return} (%)", 'font_color': '#F18F01'},
                            number={"font": {"size": 50}, 'font_color': '#F18F01', "suffix": "%"},
                            domain={'row': 2, 'column': 1}
                        ))
                        
                        fig_kpi.update_layout(
                            grid={'rows': 3, 'columns': 4, 'pattern': "independent"},
                            autosize=False,
                            width=900,
                            height=550,
                            margin=dict(l=20, r=0, b=50, t=50, pad=4)
                        )
                        
                        st.plotly_chart(fig_kpi, use_container_width=True)
                        
                        # Return Plots with Bands
                        plot_title = "📈 Rolling Return Analysis with Bands" if analysis_type == "Rolling Returns" else "📈 Return Analysis with Bands"
                        st.subheader(plot_title)
                        
                        # Primary Return Type Highlight Plot
                        fig_primary = go.Figure()
                        
                        if analysis_type == "Rolling Returns":
                            # Add rolling returns plot
                            valid_indices = ~primary_rolling_data['mean'].isna()
                            fig_primary.add_trace(go.Scatter(
                                x=timeframe_data.loc[valid_indices, 'year_numeric'],
                                y=primary_rolling_data['mean'][valid_indices],
                                mode='lines+markers',
                                name=f'{selected_return} {rolling_period}-Year Rolling Returns',
                                line=dict(color='#A23B72', width=3),
                                marker=dict(size=6)
                            ))
                            
                            # Add secondary return type if enabled
                            if show_secondary:
                                valid_indices_sec = ~secondary_rolling_data['mean'].isna()
                                fig_primary.add_trace(go.Scatter(
                                    x=timeframe_data.loc[valid_indices_sec, 'year_numeric'],
                                    y=secondary_rolling_data['mean'][valid_indices_sec],
                                    mode='lines+markers',
                                    name=f'{secondary_return} {rolling_period}-Year Rolling Returns',
                                    line=dict(color='#2E86AB', width=2),
                                    marker=dict(size=4, opacity=0.2),
                                    opacity=0.2
                                ))
                        else:
                            # Add scatter plots for actual annual returns
                            fig_primary.add_trace(go.Scatter(
                                x=timeframe_data['year_numeric'],
                                y=timeframe_data[selected_return],
                                mode='markers',
                                name=f'{selected_return} Annual Returns',
                                marker=dict(color='#A23B72', size=8)
                            ))
                            
                            # Add secondary return type if enabled
                            if show_secondary:
                                fig_primary.add_trace(go.Scatter(
                                    x=timeframe_data['year_numeric'],
                                    y=timeframe_data[secondary_return],
                                    mode='markers',
                                    name=f'{secondary_return} Annual Returns',
                                    marker=dict(color='#2E86AB', size=6, opacity=0.2)
                                ))
                        
                        # Add horizontal lines for primary return type
                        if show_statistical_bands:
                            fig_primary.add_hline(y=primary_upper, line_dash="dash", line_color="#A23B72")
                            fig_primary.add_hline(y=primary_lower, line_dash="dash", line_color="#A23B72")
                            fig_primary.add_hline(y=primary_mean, line_dash="solid", line_color="#A23B72")
                            
                            # Add horizontal lines for secondary return type
                            if show_secondary:
                                fig_primary.add_hline(y=secondary_mean, line_dash="solid", line_color="#2E86AB", opacity=0.2)
                                fig_primary.add_hline(y=secondary_upper, line_dash="dash", line_color="#2E86AB", opacity=0.2)
                                fig_primary.add_hline(y=secondary_lower, line_dash="dash", line_color="#2E86AB", opacity=0.2)
                        
                        fig_primary.update_layout(
                            title=f'Timeline of {selected_country} Returns - {selected_return} Focus',
                            xaxis_title='Year',
                            yaxis_title='Return Rate',
                            height=600,
                            width=800,  # About 2/3 of typical screen width
                            showlegend=True,
                            margin=dict(l=50, r=50, t=80, b=50)
                        )
                        
                        # Remove gridlines
                        fig_primary.update_xaxes(showgrid=False)
                        fig_primary.update_yaxes(showgrid=False)
                        
                        st.plotly_chart(fig_primary, use_container_width=True)
                        
                        # Secondary Return Type Highlight Plot
                        if show_secondary:
                            fig_secondary = go.Figure()
                            
                            if analysis_type == "Rolling Returns":
                                # Add rolling returns plot for secondary focus
                                valid_indices_primary = ~primary_rolling_data['mean'].isna()
                                fig_secondary.add_trace(go.Scatter(
                                    x=timeframe_data.loc[valid_indices_primary, 'year_numeric'],
                                    y=primary_rolling_data['mean'][valid_indices_primary],
                                    mode='lines+markers',
                                    name=f'{selected_return} {rolling_period}-Year Rolling Returns',
                                    line=dict(color='#A23B72', width=2),
                                    marker=dict(size=4, opacity=0.2),
                                    opacity=0.2
                                ))
                                
                                valid_indices_secondary = ~secondary_rolling_data['mean'].isna()
                                fig_secondary.add_trace(go.Scatter(
                                    x=timeframe_data.loc[valid_indices_secondary, 'year_numeric'],
                                    y=secondary_rolling_data['mean'][valid_indices_secondary],
                                    mode='lines+markers',
                                    name=f'{secondary_return} {rolling_period}-Year Rolling Returns',
                                    line=dict(color='#2E86AB', width=3),
                                    marker=dict(size=6)
                                ))
                            else:
                                # Add scatter plots for actual annual returns
                                fig_secondary.add_trace(go.Scatter(
                                    x=timeframe_data['year_numeric'],
                                    y=timeframe_data[selected_return],
                                    mode='markers',
                                    name=f'{selected_return} Annual Returns',
                                    marker=dict(color='#A23B72', size=6, opacity=0.2)
                                ))
                                
                                fig_secondary.add_trace(go.Scatter(
                                    x=timeframe_data['year_numeric'],
                                    y=timeframe_data[secondary_return],
                                    mode='markers',
                                    name=f'{secondary_return} Annual Returns',
                                    marker=dict(color='#2E86AB', size=8)
                                ))
                            
                            # Add horizontal lines for primary return type (faded)
                            if show_statistical_bands:
                                fig_secondary.add_hline(y=primary_upper, line_dash="dash", line_color="#A23B72", 
                                                    line_width=1, opacity=0.2)
                                fig_secondary.add_hline(y=primary_lower, line_dash="dash", line_color="#A23B72", 
                                                    line_width=1, opacity=0.2)
                                fig_secondary.add_hline(y=primary_mean, line_dash="solid", line_color="#A23B72", 
                                                    line_width=1, opacity=0.2)
                            
                            # Add horizontal lines for secondary return type (highlighted)
                            if show_statistical_bands:
                                fig_secondary.add_hline(y=secondary_mean, line_dash="solid", line_color="#2E86AB")
                                fig_secondary.add_hline(y=secondary_upper, line_dash="dash", line_color="#2E86AB")
                                fig_secondary.add_hline(y=secondary_lower, line_dash="dash", line_color="#2E86AB")
                            
                            fig_secondary.update_layout(
                                title=f'Timeline of {selected_country} Returns - {secondary_return} Focus',
                                xaxis_title='Year',
                                yaxis_title='Return Rate',
                                height=600,
                                width=800,  # About 2/3 of typical screen width
                                showlegend=True,
                                margin=dict(l=50, r=50, t=80, b=50)
                            )
                            
                            # Remove gridlines
                            fig_secondary.update_xaxes(showgrid=False)
                            fig_secondary.update_yaxes(showgrid=False)
                            
                            st.plotly_chart(fig_secondary, use_container_width=True)
                        
                        # Additional Statistics
                        stats_title = f"📊 Rolling Statistics ({rolling_period}-Year)" if analysis_type == "Rolling Returns" else "📊 Additional Statistics"
                        st.subheader(stats_title)
                        
                        col1, col2 = st.columns(2)
                        
                        analysis_suffix = f" ({rolling_period}-Year Rolling)" if analysis_type == "Rolling Returns" else " (Annual)"
                        
                        with col1:
                            st.write(f"**{selected_return} Analysis{analysis_suffix}:**")
                            if analysis_type == "Rolling Returns":
                                st.write(f"- Geometric Mean Rolling Return: {primary_mean:.4f} ({primary_mean*100:.2f}%)")
                                st.write(f"- Average Total Return per {rolling_period}-Year Period: {primary_total_return*100:.2f}%")
                                st.write(f"- Average Rolling Std Dev: {primary_std:.4f} ({primary_std*100:.2f}%)")
                                st.write(f"- Average Negative Periods: {primary_pc_negative:.1f}%")
                            else:
                                st.write(f"- Mean Return: {primary_mean:.4f} ({primary_mean*100:.2f}%)")
                                st.write(f"- Standard Deviation: {primary_std:.4f} ({primary_std*100:.2f}%)")
                                st.write(f"- Negative Years: {primary_pc_negative:.1f}%")
                            st.write(f"- Sharpe Ratio: {primary_mean/primary_std:.4f}" if primary_std != 0 else "- Sharpe Ratio: N/A")
                        
                        with col2:
                            st.write(f"**{secondary_return} Analysis{analysis_suffix}:**")
                            if analysis_type == "Rolling Returns":
                                st.write(f"- Geometric Mean Rolling Return: {secondary_mean:.4f} ({secondary_mean*100:.2f}%)")
                                st.write(f"- Average Total Return per {rolling_period}-Year Period: {secondary_total_return*100:.2f}%")
                                st.write(f"- Average Rolling Std Dev: {secondary_std:.4f} ({secondary_std*100:.2f}%)")
                                st.write(f"- Average Negative Periods: {secondary_pc_negative:.1f}%")
                            else:
                                st.write(f"- Mean Return: {secondary_mean:.4f} ({secondary_mean*100:.2f}%)")
                                st.write(f"- Standard Deviation: {secondary_std:.4f} ({secondary_std*100:.2f}%)")
                                st.write(f"- Negative Years: {secondary_pc_negative:.1f}%")
                            st.write(f"- Sharpe Ratio: {secondary_mean/secondary_std:.4f}" if secondary_std != 0 else "- Sharpe Ratio: N/A")
                    
                else:
                    st.warning(f"Could not find return columns for analysis")
            else:
                st.warning(f"No data found for {selected_country}")
        else:
            st.warning("No 'country' column found in the dataset. Please check your data structure.")
            st.info("Expected columns: 'country', return columns (e.g., 'eq_tr', 'housing_tr'), 'year'")
    
    # Data Source and Methodology Section
    st.markdown("---")
    st.header("📚 Data Source & Methodology")
    
    st.markdown("""
    ### Citation
    
    This dashboard uses data from:
    
    **Jordà, Óscar, Katharina Knoll, Dmitry Kuvshinov, Moritz Schularick, and Alan M. Taylor. 2017. "The Rate of Return on Everything, 1870–2015."** 
    Federal Reserve Bank of San Francisco Working Paper 2017-25. 
    [https://www.frbsf.org/wp-content/uploads/wp2017-25.pdf](https://www.frbsf.org/wp-content/uploads/wp2017-25.pdf)
    
    ### Key Findings from the Paper
    
    The paper presents the first comprehensive dataset on total rates of return for all major asset classes across 16 advanced economies from 1870 to 2015. Key findings include:
    
    - **Housing Returns**: Housing has delivered returns comparable to equities over the long run, with lower volatility. This is significant given that housing represents roughly half of national wealth in typical economies.
    
    - **Risk-Return Trade-off**: Equities and housing (risky assets) have historically outperformed bonds and bills (safe assets), but with higher volatility and more frequent negative periods.
    
    - **Long-term Trends**: Real returns have been relatively stable over the 145-year period, with some variation across countries and time periods.
    
    - **Return Components**: Total returns consist of two components: **investment income** (yields, dividends, rents) and **capital gains** (price changes).
    
    ### Return Type Definitions
    
    The dashboard displays various return metrics. Here are the definitions of the return type fields:
    
    - **`eq_tr`** (Equity Total Return): Total return on equities, including both dividend income and capital gains from stock price appreciation.
    
    - **`housing_tr`** (Housing Total Return): Total return on residential real estate, including rental income (yield) and capital gains from house price appreciation. This is the first comprehensive dataset to include housing returns.
    
    - **`bond_tr`** (Bond Total Return): Total return on long-term government bonds, including coupon payments and capital gains/losses from bond price changes.
    
    - **`bill_tr`** (Bill Total Return): Total return on short-term government bills (typically 3-month treasury bills), including interest income.
    
    - **`risky_tr`** (Risky Asset Total Return): A composite measure combining returns on risky assets (equities and housing).
    
    - **`safe_tr`** (Safe Asset Total Return): A composite measure combining returns on safe assets (bonds and bills).
    
    All returns are calculated in **real terms** (adjusted for inflation) and represent annual total returns, which combine investment income (yields, dividends, rents) with capital gains (price changes).
    
    ### Methodology Notes
    
    - Returns are calculated from actual market data, not inferred from wealth estimates
    - Data covers 16 advanced economies: Australia, Belgium, Denmark, Finland, France, Germany, Italy, Japan, Netherlands, Norway, Spain, Sweden, Switzerland, United Kingdom, and United States
    - The dataset addresses historical market discontinuities (e.g., war-related closures) and uses original sources for maximum accuracy
    - Housing returns are constructed from house price indices and rental yield data
    
    For detailed methodology and data sources, please refer to the original working paper.
    """)

if __name__ == "__main__":
    main()
