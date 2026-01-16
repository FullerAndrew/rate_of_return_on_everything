import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import pyreadstat for Stata file reading
try:
    import pyreadstat
    STATA_READER = 'pyreadstat'
except ImportError:
    try:
        import pandas.io.stata
        STATA_READER = 'pandas'
    except ImportError:
        STATA_READER = None

import os

# Page configuration
st.set_page_config(
    page_title="Rate of Return on Everything Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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
    /* Color KPI metric values */
    .metric-primary [data-testid="stMetricValue"] {
        color: #A23B72 !important;
    }
    .metric-secondary [data-testid="stMetricValue"] {
        color: #2E86AB !important;
    }
    .metric-comparison [data-testid="stMetricValue"] {
        color: #F18F01 !important;
    }
</style>
""", unsafe_allow_html=True)

def debug_file_paths():
    """Debug function to show current working directory and available files"""
    st.sidebar.header("🐛 Debug Information")
    
    # Show current working directory
    cwd = os.getcwd()
    st.sidebar.write(f"**Current working directory:** {cwd}")
    
    # List files in current directory
    st.sidebar.write("**Files in current directory:**")
    try:
        files = os.listdir(".")
        for file in sorted(files):
            if os.path.isfile(file):
                st.sidebar.write(f"📄 {file}")
            else:
                st.sidebar.write(f"📁 {file}/")
    except Exception as e:
        st.sidebar.write(f"Error listing files: {e}")
    
    # Check specific paths
    st.sidebar.write("**Checking specific paths:**")
    paths_to_check = [
        "./6389799/RORE_QJE_replication_v2/data/rore_public_main.dta",
        "6389799/RORE_QJE_replication_v2/data/rore_public_main.dta",
        "../6389799/RORE_QJE_replication_v2/data/rore_public_main.dta"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            st.sidebar.write(f"✅ {path} - EXISTS")
        else:
            st.sidebar.write(f"❌ {path} - NOT FOUND")

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

def parse_return_columns(return_cols):
    """Parse return column names to extract asset classes and return types
    
    Args:
        return_cols: List of column names like 'eq_tr', 'housing_tr', 'bond_capgain', etc.
    
    Returns:
        dict with 'asset_classes' (list), 'return_types' (list), and 'mapping' (dict mapping (asset, return) -> column_name)
    """
    asset_classes = set()
    return_types = set()
    mapping = {}  # (asset_class, return_type) -> column_name
    
    # Common asset class prefixes (order matters - check longer ones first)
    asset_prefixes = ['housing', 'risky', 'safe', 'eq', 'bond', 'bill']
    # Common return type suffixes
    return_suffixes = ['tr', 'capgain', 'rent', 'div', 'dp', 'yd', 'rate', 'rtn']
    
    for col in return_cols:
        col_lower = col.lower()
        # Skip excluded columns (should already be filtered, but double-check)
        if 'eq_tr_interp' in col_lower or 'capital_tr' in col_lower or '_interp' in col_lower:
            continue
            
        # Try to match asset class and return type
        matched = False
        for asset in asset_prefixes:
            prefix = asset + '_'
            if col_lower.startswith(prefix):
                # Extract return type after the underscore
                remaining = col_lower[len(prefix):]
                # Check if remaining matches a return type exactly (preferred - this catches eq_tr, housing_tr, etc.)
                if remaining in return_suffixes:
                    asset_classes.add(asset)
                    return_types.add(remaining)
                    # Only map if we don't already have a mapping for this combination (prefer first/exact match)
                    if (asset, remaining) not in mapping:
                        mapping[(asset, remaining)] = col
                    matched = True
                    break
                # Also check if it starts with a return type (for cases like 'rent_yd')
                for ret_type in return_suffixes:
                    if remaining.startswith(ret_type):
                        asset_classes.add(asset)
                        return_types.add(ret_type)
                        # Prefer exact matches over partial matches - only update if no exact match exists
                        if (asset, ret_type) not in mapping:
                            mapping[(asset, ret_type)] = col
                        matched = True
                        break
                if matched:
                    break
        
        # If no match found, try to infer from column name structure
        if not matched and '_' in col_lower:
            parts = col_lower.split('_')
            if len(parts) >= 2:
                # Assume first part is asset, last part is return type
                potential_asset = parts[0]
                potential_return = parts[-1]
                if potential_return in return_suffixes:
                    asset_classes.add(potential_asset)
                    return_types.add(potential_return)
                    mapping[(potential_asset, potential_return)] = col
    
    return {
        'asset_classes': sorted(list(asset_classes)),
        'return_types': sorted(list(return_types)),
        'mapping': mapping
    }

def create_return_histogram(returns_data, bins=None):
    """Create histogram data for return distribution
    
    Args:
        returns_data: Series or array of return values (as decimals, e.g., 0.15 for 15%)
        bins: Optional list of bin edges. If None, creates default bins with 5% increments
    
    Returns:
        dict with 'bins', 'counts', 'labels', 'colors' for plotting
    """
    # Convert to percentage for binning
    returns_pct = returns_data * 100
    
    # Create bins similar to the image: 5% increments
    if bins is None:
        min_return = returns_pct.min()
        max_return = returns_pct.max()
        # Round to nearest 5% for bin edges, extend range slightly
        min_bin = (np.floor(min_return / 5) * 5) - 5
        max_bin = (np.ceil(max_return / 5) * 5) + 5
        bins = np.arange(min_bin, max_bin + 5, 5)
    
    # Calculate histogram
    counts, bin_edges = np.histogram(returns_pct, bins=bins)
    
    # Create bin labels and colors
    bin_labels = []
    colors = []
    for i in range(len(counts)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Create label similar to image format
        if bin_start < -40:
            label = f"<{bin_start:.0f}%"
        elif bin_end > 60:
            label = f">{bin_end:.0f}%"
        else:
            label = f"{bin_start:.0f}% to {bin_end:.0f}%"
        
        bin_labels.append(label)
        
        # Color: muted red for negative, muted blue for positive (based on bin center)
        bin_center = (bin_start + bin_end) / 2
        if bin_center < 0:
            colors.append('#c85a5a')  # Muted red for negative
        else:
            colors.append('#5a8fc8')  # Muted blue for positive
    
    return {
        'bins': bin_edges,
        'counts': counts,
        'labels': bin_labels,
        'colors': colors
    }

@st.cache_data
def load_data():
    """Load the Rate of Return on Everything dataset"""
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
            if STATA_READER == 'pyreadstat':
                df, meta = pyreadstat.read_dta(path)
            else:  # pandas
                df = pd.read_stata(path)
            
            # Preprocess the data to handle common Stata issues
            df = preprocess_stata_data(df)
            
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error with path {path}: {str(e)}")
            continue
    
    # If none of the paths work, show error
    st.error("Data file not found. Please ensure the Stata file is in the correct location.")
    st.info("Expected file: rore_public_main.dta")
    st.info("Available paths tried:")
    for path in possible_paths:
        st.info(f"  - {path}")
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
        
        # Country selection - separate for primary and secondary
        if 'country' in df.columns:
            countries = sorted(df['country'].unique())
            default_country_idx = countries.index('AUS') if 'AUS' in countries else 0
            
            # Primary country and return selection
            st.subheader("📊 Primary Return")
            primary_country = st.selectbox("Primary Country:", countries, index=default_country_idx, key="primary_country")
            
            # Filter data for primary country
            primary_country_data = df[df['country'] == primary_country].copy()
            
            if len(primary_country_data) > 0:
                # Find all return columns for the primary country (numeric only)
                potential_cols_primary = [col for col in primary_country_data.columns if any(keyword in col.lower() for keyword in ['tr', 'return', 'ret', 'capgain', 'rent', 'div', 'dp', 'yd', 'rate', 'rtn'])]
                
                # Exclude specific columns that shouldn't be available
                excluded_cols = ['eq_tr_interp', 'capital_tr']
                potential_cols_primary = [col for col in potential_cols_primary if col not in excluded_cols]
                
                # Filter to only numeric columns
                primary_return_cols = []
                for col in potential_cols_primary:
                    try:
                        # Try to convert to numeric, if successful, it's a valid return column
                        pd.to_numeric(primary_country_data[col], errors='raise')
                        primary_return_cols.append(col)
                    except (ValueError, TypeError):
                        # Skip non-numeric columns
                        continue
                
                if len(primary_return_cols) > 0:
                    # Parse columns to get asset classes and return types
                    parsed = parse_return_columns(primary_return_cols)
                    primary_asset_classes = parsed['asset_classes']
                    primary_return_types = parsed['return_types']
                    primary_mapping = parsed['mapping']
                    
                    if len(primary_asset_classes) > 0 and len(primary_return_types) > 0:
                        # Default to 'eq' asset class
                        default_asset_idx = 0
                        if 'eq' in primary_asset_classes:
                            default_asset_idx = primary_asset_classes.index('eq')
                        primary_asset_class = st.selectbox("Primary Asset Class:", primary_asset_classes, index=default_asset_idx, key="primary_asset")
                        
                        # Filter return types available for selected asset class
                        available_return_types = [rt for rt in primary_return_types if (primary_asset_class, rt) in primary_mapping]
                        
                        if len(available_return_types) > 0:
                            # Default to 'tr' return type
                            default_return_idx = 0
                            if 'tr' in available_return_types:
                                default_return_idx = available_return_types.index('tr')
                            primary_return_type = st.selectbox("Primary Return Type:", available_return_types, index=default_return_idx, key="primary_return_type")
                            
                            # Get the actual column name
                            selected_return = primary_mapping.get((primary_asset_class, primary_return_type))
                            if selected_return is None:
                                st.warning(f"Column not found for {primary_asset_class}_{primary_return_type}")
                                st.stop()
                        else:
                            st.warning(f"No return types available for {primary_asset_class}")
                            st.stop()
                    else:
                        # Fallback to old method if parsing fails
                        default_primary_idx = 0
                        if 'eq_tr' in primary_return_cols:
                            default_primary_idx = primary_return_cols.index('eq_tr')
                        selected_return = st.selectbox("Primary Return Type:", primary_return_cols, index=default_primary_idx, key="primary_return")
                else:
                    st.warning("No valid return columns found for primary country")
                    st.stop()
            else:
                st.warning(f"No data found for {primary_country}")
                st.stop()
            
            # Secondary country and return selection
            st.subheader("📊 Secondary Return")
            secondary_country = st.selectbox("Secondary Country:", countries, index=default_country_idx, key="secondary_country")
            
            # Filter data for secondary country
            secondary_country_data = df[df['country'] == secondary_country].copy()
            
            if len(secondary_country_data) > 0:
                # Find all return columns for the secondary country (numeric only)
                potential_cols_secondary = [col for col in secondary_country_data.columns if any(keyword in col.lower() for keyword in ['tr', 'return', 'ret', 'capgain', 'rent', 'div', 'dp', 'yd', 'rate', 'rtn'])]
                
                # Exclude specific columns that shouldn't be available
                excluded_cols = ['eq_tr_interp', 'capital_tr']
                potential_cols_secondary = [col for col in potential_cols_secondary if col not in excluded_cols]
                
                # Filter to only numeric columns
                secondary_return_cols = []
                for col in potential_cols_secondary:
                    try:
                        # Try to convert to numeric, if successful, it's a valid return column
                        pd.to_numeric(secondary_country_data[col], errors='raise')
                        secondary_return_cols.append(col)
                    except (ValueError, TypeError):
                        # Skip non-numeric columns
                        continue
                
                if len(secondary_return_cols) > 0:
                    # Parse columns to get asset classes and return types
                    parsed = parse_return_columns(secondary_return_cols)
                    secondary_asset_classes = parsed['asset_classes']
                    secondary_return_types = parsed['return_types']
                    secondary_mapping = parsed['mapping']
                    
                    if len(secondary_asset_classes) > 0 and len(secondary_return_types) > 0:
                        # Default to 'housing' asset class
                        default_asset_idx = 0
                        if 'housing' in secondary_asset_classes:
                            default_asset_idx = secondary_asset_classes.index('housing')
                        secondary_asset_class = st.selectbox("Secondary Asset Class:", secondary_asset_classes, index=default_asset_idx, key="secondary_asset")
                        
                        # Filter return types available for selected asset class
                        available_return_types = [rt for rt in secondary_return_types if (secondary_asset_class, rt) in secondary_mapping]
                        
                        if len(available_return_types) > 0:
                            # Default to 'tr' return type
                            default_return_idx = 0
                            if 'tr' in available_return_types:
                                default_return_idx = available_return_types.index('tr')
                            secondary_return_type = st.selectbox("Secondary Return Type:", available_return_types, index=default_return_idx, key="secondary_return_type")
                            
                            # Get the actual column name
                            secondary_return = secondary_mapping.get((secondary_asset_class, secondary_return_type))
                            if secondary_return is None:
                                st.warning(f"Column not found for {secondary_asset_class}_{secondary_return_type}")
                                st.stop()
                        else:
                            st.warning(f"No return types available for {secondary_asset_class}")
                            st.stop()
                    else:
                        # Fallback to old method if parsing fails
                        default_secondary_idx = 0
                        if 'housing_tr' in secondary_return_cols:
                            default_secondary_idx = secondary_return_cols.index('housing_tr')
                        secondary_return = st.selectbox("Secondary Return Type:", secondary_return_cols, index=default_secondary_idx, key="secondary_return")
                else:
                    st.warning("No valid return columns found for secondary country")
                    st.stop()
            else:
                st.warning(f"No data found for {secondary_country}")
                st.stop()
            
            # Use primary country for display purposes (for compatibility with existing code)
            selected_country = primary_country
            country_data = primary_country_data
            
            # Timeframe selection - find common years between both countries
            st.subheader("⏰ Timeframe")
            if 'year_numeric' in primary_country_data.columns and 'year_numeric' in secondary_country_data.columns:
                primary_years = set(primary_country_data['year_numeric'].unique())
                secondary_years = set(secondary_country_data['year_numeric'].unique())
                common_years = sorted(list(primary_years.intersection(secondary_years)))
                
                if len(common_years) > 1:
                    # Find index for 1951, default to 0 if not found
                    default_start_index = 0
                    if 1951 in common_years:
                        default_start_index = common_years.index(1951)
                    start_year = st.selectbox("Start Year:", common_years, index=default_start_index)
                    end_year = st.selectbox("End Year:", common_years, index=len(common_years)-1)
                    
                    # Filter data by timeframe for both countries
                    primary_timeframe = primary_country_data[
                        (primary_country_data['year_numeric'] >= start_year) & 
                        (primary_country_data['year_numeric'] <= end_year)
                    ].copy()
                    
                    secondary_timeframe = secondary_country_data[
                        (secondary_country_data['year_numeric'] >= start_year) & 
                        (secondary_country_data['year_numeric'] <= end_year)
                    ].copy()
                    
                    # Merge primary and secondary data on year_numeric
                    timeframe_data = primary_timeframe.merge(
                        secondary_timeframe[['year_numeric', secondary_return]].rename(columns={secondary_return: f'secondary_{secondary_return}'}),
                        on='year_numeric',
                        how='inner'
                    )
                    
                    # Update secondary_return to point to the merged column
                    secondary_return_merged = f'secondary_{secondary_return}'
                else:
                    st.warning(f"No common years found between {primary_country} and {secondary_country}")
                    st.stop()
            else:
                st.warning("Year column not found in data")
                st.stop()
            
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
            focus_return = st.radio("Focus Return Type", ["Primary", "Secondary"], index=0)
            show_secondary_overlay = st.checkbox("Show Secondary Return Overlay", value=True)
            show_histogram_bars = st.checkbox("Show Histogram Bars", value=True)
            show_statistical_bands = st.checkbox("Show Statistical Bands", value=True)
            
            # Set opacity based on focus
            if focus_return == "Primary":
                primary_opacity = 1.0
                secondary_opacity = 0.2
            else:  # Secondary focus
                primary_opacity = 0.2
                secondary_opacity = 1.0
            
            # Show what columns we found
            st.info(f"**Primary:** {primary_country} - {selected_return}")
            st.info(f"**Secondary:** {secondary_country} - {secondary_return}")
            st.info(f"**Timeframe:** {start_year} - {end_year}")
            st.info(f"**Common years:** {len(timeframe_data)} years")
            
            if len(timeframe_data) > 0:
                try:
                    # Ensure columns are numeric before calculating statistics
                    timeframe_data[selected_return] = pd.to_numeric(timeframe_data[selected_return], errors='coerce')
                    timeframe_data[secondary_return_merged] = pd.to_numeric(timeframe_data[secondary_return_merged], errors='coerce')
                    
                    # Use the merged secondary column for calculations
                    secondary_return_calc = secondary_return_merged
                    
                    # Sort by year to ensure proper rolling calculations
                    if 'year_numeric' in timeframe_data.columns:
                        timeframe_data = timeframe_data.sort_values('year_numeric').reset_index(drop=True)
                    
                    if analysis_type == "Rolling Returns":
                        # Calculate rolling statistics
                        primary_rolling = calculate_rolling_statistics(timeframe_data[selected_return], rolling_period)
                        secondary_rolling = calculate_rolling_statistics(timeframe_data[secondary_return_calc], rolling_period)
                        
                        # Use rolling statistics for KPIs
                        primary_mean = primary_rolling['mean'].mean()
                        primary_median = primary_rolling['mean'].median()
                        primary_total_return = primary_rolling['total_return'].mean()
                        primary_std = primary_rolling['std'].mean()
                        primary_upper = primary_mean + primary_std
                        primary_lower = primary_mean - primary_std
                        primary_pc_negative = primary_rolling['negative_pct'].iloc[0]
                        
                        secondary_mean = secondary_rolling['mean'].mean()
                        secondary_median = secondary_rolling['mean'].median()
                        secondary_total_return = secondary_rolling['total_return'].mean()
                        secondary_std = secondary_rolling['std'].mean()
                        secondary_upper = secondary_mean + secondary_std
                        secondary_lower = secondary_mean - secondary_std
                        secondary_pc_negative = secondary_rolling['negative_pct'].iloc[0]
                        
                        # Calculate number of rolling periods where asset 1 beats asset 2
                        primary_rolling_means = primary_rolling['mean'].dropna()
                        secondary_rolling_means = secondary_rolling['mean'].dropna()
                        common_indices = primary_rolling_means.index.intersection(secondary_rolling_means.index)
                        if len(common_indices) > 0:
                            asset1_beats_asset2_count = (primary_rolling_means.loc[common_indices] > secondary_rolling_means.loc[common_indices]).sum()
                            total_comparable_periods = len(common_indices)
                            asset1_beats_asset2_pct = (asset1_beats_asset2_count / total_comparable_periods * 100) if total_comparable_periods > 0 else 0
                        else:
                            asset1_beats_asset2_count = 0
                            total_comparable_periods = 0
                            asset1_beats_asset2_pct = 0
                        
                        # Calculate Sharpe ratios
                        primary_sharpe = (primary_mean / primary_std) if primary_std > 0 else np.nan
                        secondary_sharpe = (secondary_mean / secondary_std) if secondary_std > 0 else np.nan
                        
                        # Store rolling data for plotting
                        primary_rolling_data = primary_rolling
                        secondary_rolling_data = secondary_rolling
                    else:
                        # Calculate annual statistics
                        primary_mean = timeframe_data[selected_return].mean()
                        primary_median = timeframe_data[selected_return].median()
                        primary_std = timeframe_data[selected_return].std()
                        primary_upper = primary_mean + primary_std
                        primary_lower = primary_mean - primary_std
                        primary_pc_negative = (timeframe_data[selected_return] < 0).mean() * 100
                        
                        secondary_mean = timeframe_data[secondary_return_calc].mean()
                        secondary_median = timeframe_data[secondary_return_calc].median()
                        secondary_std = timeframe_data[secondary_return_calc].std()
                        secondary_upper = secondary_mean + secondary_std
                        secondary_lower = secondary_mean - secondary_std
                        secondary_pc_negative = (timeframe_data[secondary_return_calc] < 0).mean() * 100
                        
                        # Calculate Sharpe ratios
                        primary_sharpe = (primary_mean / primary_std) if primary_std > 0 else np.nan
                        secondary_sharpe = (secondary_mean / secondary_std) if secondary_std > 0 else np.nan
                        
                        # Calculate number of annual periods where asset 1 beats asset 2
                        asset1_beats_asset2_count = (timeframe_data[selected_return] > timeframe_data[secondary_return_calc]).sum()
                        total_comparable_periods = len(timeframe_data)
                        asset1_beats_asset2_pct = (asset1_beats_asset2_count / total_comparable_periods * 100) if total_comparable_periods > 0 else 0
                        
                        # Set rolling data to None for annual analysis
                        primary_rolling_data = None
                        secondary_rolling_data = None
                    
                except Exception as e:
                    st.error(f"Error calculating statistics: {e}")
                    st.error(f"Selected return column '{selected_return}' may contain non-numeric data")
                    st.error(f"Secondary return column '{secondary_return_calc}' may contain non-numeric data")
                    st.stop()
                
                # Create KPI Widgets using Streamlit native metrics
                with col_left:
                        st.subheader("Portfolio Statistics")
                        
                        # Combined metrics showing both assets side by side in 5 columns across top
                        col1, col2, col3, col4, col5 = st.columns(5)
                        period_label = f"{rolling_period}-Year Rolling Periods" if analysis_type == "Rolling Returns" else "Annual Periods"
                        
                        # Column 1: Mean
                        with col1:
                            st.markdown('<div class="metric-primary">', unsafe_allow_html=True)
                            st.metric(
                                f"Mean ({selected_return} / {secondary_return})",
                                f"{primary_mean*100:.2f}% / {secondary_mean*100:.2f}%",
                                help=f"Average return: {selected_return} / {secondary_return}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Column 2: Negative Years/Periods
                        with col2:
                            st.markdown('<div class="metric-primary">', unsafe_allow_html=True)
                            if analysis_type == "Rolling Returns":
                                st.metric(
                                    f"Negative Periods ({selected_return} / {secondary_return})",
                                    f"{primary_pc_negative:.1f}% / {secondary_pc_negative:.1f}%",
                                    help=f"Percentage of {rolling_period}-year rolling periods with negative returns"
                                )
                            else:
                                st.metric(
                                    f"Negative Years ({selected_return} / {secondary_return})",
                                    f"{primary_pc_negative:.1f}% / {secondary_pc_negative:.1f}%",
                                    help=f"Percentage of years with negative returns"
                                )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Column 3: Beats
                        with col3:
                            st.markdown('<div class="metric-comparison">', unsafe_allow_html=True)
                            st.metric(
                                f"{selected_return} Beats {secondary_return}",
                                f"{asset1_beats_asset2_count} / {total_comparable_periods}",
                                f"{asset1_beats_asset2_pct:.1f}%",
                                help=f"Number and percentage of {period_label.lower()} where {selected_return} outperforms {secondary_return}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Column 4: Std Dev
                        with col4:
                            st.markdown('<div class="metric-primary">', unsafe_allow_html=True)
                            st.metric(
                                f"Std Dev ({selected_return} / {secondary_return})",
                                f"{primary_std*100:.2f}% / {secondary_std*100:.2f}%",
                                help=f"Standard deviation: {selected_return} / {secondary_return}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Column 5: Sharpe Ratio
                        with col5:
                            st.markdown('<div class="metric-primary">', unsafe_allow_html=True)
                            primary_sharpe_str = f"{primary_sharpe:.2f}" if not np.isnan(primary_sharpe) else "N/A"
                            secondary_sharpe_str = f"{secondary_sharpe:.2f}" if not np.isnan(secondary_sharpe) else "N/A"
                            st.metric(
                                f"Sharpe Ratio ({selected_return} / {secondary_return})",
                                f"{primary_sharpe_str} / {secondary_sharpe_str}",
                                help=f"Sharpe ratio (mean/std): {selected_return} / {secondary_return}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Return Plots with Bands and Histogram
                        plot_title = "📈 Rolling Return Analysis with Bands" if analysis_type == "Rolling Returns" else "📈 Return Analysis with Bands"
                        st.subheader(plot_title)
                        
                        # Create histogram data for primary return
                        if analysis_type == "Rolling Returns":
                            primary_returns_for_hist = primary_rolling_data['mean'].dropna()
                        else:
                            primary_returns_for_hist = timeframe_data[selected_return].dropna()
                        
                        # Always create secondary histogram data (for overlay and shared axis calculation)
                        if analysis_type == "Rolling Returns":
                            secondary_returns_for_hist = secondary_rolling_data['mean'].dropna()
                        else:
                            secondary_returns_for_hist = timeframe_data[secondary_return_calc].dropna()
                        
                        # Calculate shared bins based on combined range of both datasets
                        combined_returns_pct = pd.concat([primary_returns_for_hist * 100, secondary_returns_for_hist * 100])
                        min_return = combined_returns_pct.min()
                        max_return = combined_returns_pct.max()
                        min_bin = (np.floor(min_return / 5) * 5) - 5
                        max_bin = (np.ceil(max_return / 5) * 5) + 5
                        shared_bins = np.arange(min_bin, max_bin + 5, 5)
                        
                        # Create histograms with shared bins
                        hist_data = create_return_histogram(primary_returns_for_hist, bins=shared_bins)
                        hist_data_secondary = create_return_histogram(secondary_returns_for_hist, bins=shared_bins)
                        
                        # Calculate shared axis ranges for histograms
                        non_zero_indices = [i for i, count in enumerate(hist_data['counts']) if count > 0]
                        hist_counts = [hist_data['counts'][i] for i in non_zero_indices]
                        total_count = sum(hist_counts)
                        hist_percentages = [(count / total_count * 100) if total_count > 0 else 0 for count in hist_counts]
                        
                        # Get all bin edges for x-axis range
                        all_bin_edges = list(hist_data['bins'])
                        all_bin_labels = hist_data['labels']
                        
                        # Always calculate secondary histogram data
                        non_zero_indices_sec = [i for i, count in enumerate(hist_data_secondary['counts']) if count > 0]
                        hist_counts_sec = [hist_data_secondary['counts'][i] for i in non_zero_indices_sec]
                        total_count_sec = sum(hist_counts_sec)
                        hist_percentages_sec = [(count / total_count_sec * 100) if total_count_sec > 0 else 0 for count in hist_counts_sec]
                        
                        # Combine bin edges and percentages for shared ranges
                        all_bin_edges.extend(hist_data_secondary['bins'])
                        all_bin_labels.extend(hist_data_secondary['labels'])
                        all_percentages = hist_percentages + hist_percentages_sec
                        
                        # Calculate shared ranges
                        min_x = min(all_bin_edges) if all_bin_edges else 0
                        max_x = max(all_bin_edges) if all_bin_edges else 100
                        max_y = max(all_percentages) * 1.1 if all_percentages else 100
                        
                        # Create subplots: main chart on left (70%), histogram on right (30%)
                        # Update title based on focus
                        focus_return_name = selected_return if focus_return == "Primary" else secondary_return
                        fig_primary = make_subplots(
                            rows=1, cols=2,
                            column_widths=[0.7, 0.3],
                            subplot_titles=(f'Timeline of {selected_country} Returns - {focus_return_name} Focus', f'Return Distribution - {focus_return_name}'),
                            horizontal_spacing=0.05,
                            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        if analysis_type == "Rolling Returns":
                            # Add rolling returns plot to left subplot
                            valid_indices = ~primary_rolling_data['mean'].isna()
                            fig_primary.add_trace(go.Scatter(
                                x=timeframe_data.loc[valid_indices, 'year_numeric'],
                                y=primary_rolling_data['mean'][valid_indices],
                                mode='lines+markers',
                                name=f'{selected_return} {rolling_period}-Year Rolling Returns',
                                line=dict(color='#A23B72', width=3),
                                marker=dict(size=6),
                                opacity=primary_opacity,
                                showlegend=False
                            ), row=1, col=1)
                            
                            # Add secondary return type as overlay if enabled
                            if show_secondary_overlay:
                                valid_indices_sec = ~secondary_rolling_data['mean'].isna()
                                fig_primary.add_trace(go.Scatter(
                                    x=timeframe_data.loc[valid_indices_sec, 'year_numeric'],
                                    y=secondary_rolling_data['mean'][valid_indices_sec],
                                    mode='lines+markers',
                                    name=f'{secondary_return} {rolling_period}-Year Rolling Returns',
                                    line=dict(color='#2E86AB', width=2),
                                    marker=dict(size=4),
                                    opacity=secondary_opacity,
                                    showlegend=False
                                ), row=1, col=1)
                        else:
                            # Add scatter plots for actual annual returns to left subplot
                            fig_primary.add_trace(go.Scatter(
                                x=timeframe_data['year_numeric'],
                                y=timeframe_data[selected_return],
                                mode='markers',
                                name=f'{selected_return} Annual Returns',
                                marker=dict(color='#A23B72', size=8),
                                opacity=primary_opacity,
                                showlegend=False
                            ), row=1, col=1)
                            
                            # Add secondary return type as overlay if enabled
                            if show_secondary_overlay:
                                fig_primary.add_trace(go.Scatter(
                                    x=timeframe_data['year_numeric'],
                                    y=timeframe_data[secondary_return_calc],
                                    mode='markers',
                                    name=f'{secondary_return} Annual Returns',
                                    marker=dict(color='#2E86AB', size=6),
                                    opacity=secondary_opacity,
                                    showlegend=False
                                ), row=1, col=1)
                        
                        # Add horizontal lines for primary return type (left subplot only)
                        if show_statistical_bands:
                            fig_primary.add_hline(y=primary_upper, line_dash="dash", line_color="#A23B72", opacity=primary_opacity, row=1, col=1)
                            fig_primary.add_hline(y=primary_lower, line_dash="dash", line_color="#A23B72", opacity=primary_opacity, row=1, col=1)
                            fig_primary.add_hline(y=primary_mean, line_dash="solid", line_color="#A23B72", opacity=primary_opacity, row=1, col=1)
                            
                            # Add horizontal lines for secondary return type as overlay if enabled
                            if show_secondary_overlay:
                                fig_primary.add_hline(y=secondary_mean, line_dash="solid", line_color="#2E86AB", opacity=secondary_opacity, row=1, col=1)
                                fig_primary.add_hline(y=secondary_upper, line_dash="dash", line_color="#2E86AB", opacity=secondary_opacity, row=1, col=1)
                                fig_primary.add_hline(y=secondary_lower, line_dash="dash", line_color="#2E86AB", opacity=secondary_opacity, row=1, col=1)
                        
                        # Add histogram to right subplot (col 2)
                        # Create complete set of bin labels from shared bins to ensure alignment
                        hist_bin_edges = hist_data['bins']
                        
                        # Create complete arrays for all bins (including zeros) to ensure x-axis alignment
                        all_hist_percentages = []
                        all_hist_counts = []
                        all_hist_colors = []
                        all_text_labels = []
                        
                        for i in range(len(hist_data['counts'])):
                            if i in non_zero_indices:
                                idx = non_zero_indices.index(i)
                                all_hist_percentages.append(hist_percentages[idx])
                                all_hist_counts.append(hist_counts[idx])
                                all_text_labels.append(f'{hist_percentages[idx]:.1f}%')
                            else:
                                all_hist_percentages.append(0)
                                all_hist_counts.append(0)
                                all_text_labels.append('')
                            
                            # Colors for all bins
                            bin_start = hist_bin_edges[i]
                            bin_end = hist_bin_edges[i + 1]
                            bin_center = (bin_start + bin_end) / 2
                            if bin_center >= 0:
                                all_hist_colors.append(f'rgba(162, 59, 114, {primary_opacity})')  # Purple
                            else:
                                all_hist_colors.append(f'rgba(200, 90, 90, {primary_opacity})')  # Red
                        
                        # Add histogram bars if enabled
                        if show_histogram_bars:
                            fig_primary.add_trace(go.Bar(
                                x=all_bin_labels,
                                y=all_hist_percentages,
                                orientation='v',
                                name='Return Distribution',
                                marker=dict(color=all_hist_colors),
                                text=all_text_labels,
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata} periods<extra></extra>',
                                customdata=all_hist_counts,
                                showlegend=False
                            ), row=1, col=2)
                        
                        # Overlay secondary distribution on primary histogram (always shown if secondary overlay is enabled)
                        if show_secondary_overlay:
                            # Create complete arrays for secondary histogram using same bin labels
                            all_hist_percentages_sec = []
                            all_hist_counts_sec = []
                            all_hist_colors_sec = []
                            
                            # Ensure we use the same number of bins as primary
                            for i in range(len(hist_data['counts'])):
                                if i < len(hist_data_secondary['counts']):
                                    if i in non_zero_indices_sec:
                                        idx = non_zero_indices_sec.index(i)
                                        all_hist_percentages_sec.append(hist_percentages_sec[idx])
                                        all_hist_counts_sec.append(hist_counts_sec[idx])
                                    else:
                                        all_hist_percentages_sec.append(0)
                                        all_hist_counts_sec.append(0)
                                else:
                                    all_hist_percentages_sec.append(0)
                                    all_hist_counts_sec.append(0)
                                
                                # Colors for secondary
                                bin_start = hist_bin_edges[i]
                                bin_end = hist_bin_edges[i + 1]
                                bin_center = (bin_start + bin_end) / 2
                                if bin_center >= 0:
                                    all_hist_colors_sec.append(f'rgba(46, 134, 171, {secondary_opacity})')  # Blue
                                else:
                                    all_hist_colors_sec.append(f'rgba(200, 90, 90, {secondary_opacity})')  # Red
                            
                            # Add secondary histogram bars with opacity based on focus
                            fig_primary.add_trace(go.Bar(
                                x=all_bin_labels,
                                y=all_hist_percentages_sec,
                                orientation='v',
                                name='Secondary Return Distribution',
                                marker=dict(color=all_hist_colors_sec),
                                hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata} periods<extra></extra>',
                                customdata=all_hist_counts_sec,
                                showlegend=False
                            ), row=1, col=2)
                            
                            # Fit normal curve to secondary histogram
                            secondary_returns_pct = secondary_returns_for_hist * 100
                            if len(secondary_returns_pct) > 0:
                                mean_secondary = secondary_returns_pct.mean()
                                std_secondary = secondary_returns_pct.std()
                                if std_secondary > 0:
                                    # Calculate normal curve values for each bin using CDF (use primary bin edges for alignment)
                                    y_norm_sampled_sec = []
                                    for i in range(len(hist_bin_edges)-1):
                                        prob = stats.norm.cdf(hist_bin_edges[i+1], mean_secondary, std_secondary) - stats.norm.cdf(hist_bin_edges[i], mean_secondary, std_secondary)
                                        y_norm_sampled_sec.append(prob * 100)
                                    
                                    # Use all bins (already calculated for primary structure)
                                    fig_primary.add_trace(go.Scatter(
                                        x=all_bin_labels,
                                        y=y_norm_sampled_sec,
                                        mode='lines',
                                        name='Secondary Normal Curve',
                                        line=dict(color='#2E86AB', width=2),
                                        opacity=secondary_opacity,
                                        showlegend=False
                                    ), row=1, col=2)
                        
                        # Fit normal curve to primary histogram
                        primary_returns_pct = primary_returns_for_hist * 100
                        if len(primary_returns_pct) > 0:
                            mean_primary = primary_returns_pct.mean()
                            std_primary = primary_returns_pct.std()
                            if std_primary > 0:
                                # Calculate normal curve values for each bin using CDF
                                y_norm_sampled = []
                                for i in range(len(hist_bin_edges)-1):
                                    prob = stats.norm.cdf(hist_bin_edges[i+1], mean_primary, std_primary) - stats.norm.cdf(hist_bin_edges[i], mean_primary, std_primary)
                                    y_norm_sampled.append(prob * 100)
                                
                                # Use all bins
                                fig_primary.add_trace(go.Scatter(
                                    x=all_bin_labels,
                                    y=y_norm_sampled,
                                    mode='lines',
                                    name='Normal Curve',
                                    line=dict(color='#A23B72', width=2),
                                    opacity=primary_opacity,
                                    showlegend=False
                                ), row=1, col=2)
                        
                        # Update layout - no legend
                        fig_primary.update_layout(
                            height=600,
                            showlegend=False,
                            margin=dict(l=50, r=50, t=80, b=50)
                        )
                        
                        # Update axes - apply shared ranges to histogram
                        fig_primary.update_xaxes(title_text='Year', showgrid=False, row=1, col=1)
                        fig_primary.update_yaxes(title_text='Return Rate', showgrid=False, row=1, col=1)
                        # Set x-axis for histogram - use category type to ensure proper label display
                        fig_primary.update_xaxes(
                            title_text='', 
                            showgrid=False, 
                            row=1, col=2, 
                            tickangle=-45,
                            type='category'
                        )
                        fig_primary.update_yaxes(title_text='', showgrid=False, row=1, col=2, range=[0, max_y])
                        
                        st.plotly_chart(fig_primary, use_container_width=True)
                        
                        # Cross-Country Comparison Table
                        st.subheader(f"📊 Cross-Country Comparison: {primary_asset_class.upper()} {primary_return_type.upper()}")
                        
                        # Get the column name for the primary asset class and return type across all countries
                        comparison_data = []
                        for country in sorted(df['country'].unique()):
                            country_df = df[df['country'] == country].copy()
                            
                            # Find the matching column for primary asset class
                            if selected_return in country_df.columns:
                                primary_col_name = selected_return
                            else:
                                # Try to find a column with the same asset class and return type
                                potential_cols = [col for col in country_df.columns 
                                                 if col.lower().startswith(f'{primary_asset_class}_') 
                                                 and primary_return_type in col.lower()]
                                if potential_cols:
                                    primary_col_name = potential_cols[0]
                                else:
                                    continue
                            
                            # Find the matching column for secondary asset class
                            if secondary_return in country_df.columns:
                                secondary_col_name = secondary_return
                            else:
                                # Try to find a column with the same asset class and return type
                                potential_cols_sec = [col for col in country_df.columns 
                                                     if col.lower().startswith(f'{secondary_asset_class}_') 
                                                     and secondary_return_type in col.lower()]
                                if potential_cols_sec:
                                    secondary_col_name = potential_cols_sec[0]
                                else:
                                    secondary_col_name = None
                            
                            # Filter by timeframe
                            country_timeframe = country_df[
                                (country_df['year_numeric'] >= start_year) & 
                                (country_df['year_numeric'] <= end_year)
                            ].copy()
                            
                            if len(country_timeframe) == 0:
                                continue
                            
                            # Ensure columns are numeric
                            country_timeframe[primary_col_name] = pd.to_numeric(country_timeframe[primary_col_name], errors='coerce')
                            primary_data_clean = country_timeframe[primary_col_name].dropna()
                            
                            if len(primary_data_clean) == 0:
                                continue
                            
                            # Calculate statistics for primary
                            if analysis_type == "Rolling Returns":
                                country_rolling = calculate_rolling_statistics(primary_data_clean, rolling_period)
                                country_mean = country_rolling['mean'].mean()
                                country_std = country_rolling['std'].mean()
                                country_negative = country_rolling['negative_pct'].iloc[0] if len(country_rolling['negative_pct']) > 0 else 0
                            else:
                                country_mean = primary_data_clean.mean()
                                country_std = primary_data_clean.std()
                                country_negative = (primary_data_clean < 0).mean() * 100
                            
                            # Calculate beats if secondary column exists
                            country_beats = "N/A"
                            if secondary_col_name and secondary_col_name in country_timeframe.columns:
                                country_timeframe[secondary_col_name] = pd.to_numeric(country_timeframe[secondary_col_name], errors='coerce')
                                secondary_data_clean = country_timeframe[secondary_col_name].dropna()
                                if len(secondary_data_clean) > 0 and len(primary_data_clean) > 0:
                                    if analysis_type == "Rolling Returns":
                                        secondary_rolling = calculate_rolling_statistics(secondary_data_clean, rolling_period)
                                        primary_means = country_rolling['mean'].dropna()
                                        secondary_means = secondary_rolling['mean'].dropna()
                                        common_indices = primary_means.index.intersection(secondary_means.index)
                                        if len(common_indices) > 0:
                                            beats_count = (primary_means.loc[common_indices] > secondary_means.loc[common_indices]).sum()
                                            total_periods = len(common_indices)
                                            country_beats_pct = (beats_count / total_periods * 100) if total_periods > 0 else 0
                                            country_beats = f"{beats_count}/{total_periods} ({country_beats_pct:.1f}%)"
                                    else:
                                        # Align the series on year_numeric for comparison
                                        primary_aligned = country_timeframe.set_index('year_numeric')[primary_col_name].dropna()
                                        secondary_aligned = country_timeframe.set_index('year_numeric')[secondary_col_name].dropna()
                                        common_years = primary_aligned.index.intersection(secondary_aligned.index)
                                        if len(common_years) > 0:
                                            beats_count = (primary_aligned.loc[common_years] > secondary_aligned.loc[common_years]).sum()
                                            total_periods = len(common_years)
                                            country_beats_pct = (beats_count / total_periods * 100) if total_periods > 0 else 0
                                            country_beats = f"{beats_count}/{total_periods} ({country_beats_pct:.1f}%)"
                                        else:
                                            country_beats = "N/A"
                            
                            # Calculate Sharpe ratio
                            country_sharpe = (country_mean / country_std) if country_std > 0 else np.nan
                            country_sharpe_str = f"{country_sharpe:.2f}" if not np.isnan(country_sharpe) else "N/A"
                            
                            comparison_data.append({
                                'Country': country,
                                'Mean (%)': f"{country_mean*100:.2f}",
                                'Negative Years (%)': f"{country_negative:.1f}" if not np.isnan(country_negative) else "N/A",
                                'Beats Secondary': country_beats,
                                'Std Dev (%)': f"{country_std*100:.2f}",
                                'Sharpe Ratio': country_sharpe_str
                            })
                        
                        if len(comparison_data) > 0:
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Convert Mean (%) to numeric for sorting (remove % and convert)
                            comparison_df['Mean_Numeric'] = comparison_df['Mean (%)'].str.replace('%', '').astype(float)
                            
                            # Sort by mean descending
                            comparison_df = comparison_df.sort_values('Mean_Numeric', ascending=False)
                            
                            # Drop the numeric column used for sorting
                            comparison_df = comparison_df.drop('Mean_Numeric', axis=1)
                            
                            # Reset index for display
                            comparison_df = comparison_df.reset_index(drop=True)
                            
                            # Create a styled dataframe to highlight the primary country
                            def highlight_country(row):
                                if row['Country'] == primary_country:
                                    return ['background-color: #fff3cd; font-weight: bold'] * len(row)
                                return [''] * len(row)
                            
                            styled_df = comparison_df.style.apply(highlight_country, axis=1)
                            
                            # Display the styled dataframe
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("No comparison data available for other countries.")
                        
                        # Country Spotlight - Show all assets for the primary country
                        st.subheader(f"🔍 Country Spotlight: {primary_country}")
                        st.markdown(f"*Key KPIs for all available assets in {primary_country}*")
                        
                        # Get all return columns for the primary country
                        country_spotlight_data = df[df['country'] == primary_country].copy()
                        
                        if len(country_spotlight_data) > 0:
                            # Find all return columns (same logic as before)
                            potential_cols_spotlight = [col for col in country_spotlight_data.columns 
                                                         if any(keyword in col.lower() for keyword in ['tr', 'return', 'ret', 'capgain', 'rent', 'div', 'dp', 'yd', 'rate', 'rtn'])]
                            
                            # Exclude specific columns
                            excluded_cols = ['eq_tr_interp', 'capital_tr']
                            potential_cols_spotlight = [col for col in potential_cols_spotlight if col not in excluded_cols]
                            
                            # Filter to only numeric columns
                            spotlight_return_cols = []
                            for col in potential_cols_spotlight:
                                try:
                                    pd.to_numeric(country_spotlight_data[col], errors='raise')
                                    spotlight_return_cols.append(col)
                                except (ValueError, TypeError):
                                    continue
                            
                            if len(spotlight_return_cols) > 0:
                                # Filter by timeframe
                                country_spotlight_timeframe = country_spotlight_data[
                                    (country_spotlight_data['year_numeric'] >= start_year) & 
                                    (country_spotlight_data['year_numeric'] <= end_year)
                                ].copy()
                                
                                if len(country_spotlight_timeframe) > 0:
                                    spotlight_data = []
                                    
                                    for asset_col in spotlight_return_cols:
                                        # Ensure column is numeric
                                        country_spotlight_timeframe[asset_col] = pd.to_numeric(
                                            country_spotlight_timeframe[asset_col], errors='coerce'
                                        )
                                        asset_data_clean = country_spotlight_timeframe[asset_col].dropna()
                                        
                                        if len(asset_data_clean) == 0:
                                            continue
                                        
                                        # Calculate statistics
                                        if analysis_type == "Rolling Returns":
                                            asset_rolling = calculate_rolling_statistics(asset_data_clean, rolling_period)
                                            asset_mean = asset_rolling['mean'].mean()
                                            asset_std = asset_rolling['std'].mean()
                                            asset_negative = asset_rolling['negative_pct'].iloc[0] if len(asset_rolling['negative_pct']) > 0 else 0
                                        else:
                                            asset_mean = asset_data_clean.mean()
                                            asset_std = asset_data_clean.std()
                                            asset_negative = (asset_data_clean < 0).mean() * 100
                                        
                                        # Calculate Sharpe ratio
                                        asset_sharpe = (asset_mean / asset_std) if asset_std > 0 else np.nan
                                        asset_sharpe_str = f"{asset_sharpe:.2f}" if not np.isnan(asset_sharpe) else "N/A"
                                        
                                        spotlight_data.append({
                                            'Asset': asset_col,
                                            'Mean (%)': f"{asset_mean*100:.2f}",
                                            'Negative Years (%)': f"{asset_negative:.1f}" if not np.isnan(asset_negative) else "N/A",
                                            'Std Dev (%)': f"{asset_std*100:.2f}",
                                            'Sharpe Ratio': asset_sharpe_str
                                        })
                                    
                                    if len(spotlight_data) > 0:
                                        spotlight_df = pd.DataFrame(spotlight_data)
                                        
                                        # Convert Mean (%) to numeric for sorting
                                        spotlight_df['Mean_Numeric'] = spotlight_df['Mean (%)'].str.replace('%', '').astype(float)
                                        
                                        # Sort by mean descending
                                        spotlight_df = spotlight_df.sort_values('Mean_Numeric', ascending=False)
                                        
                                        # Drop the numeric column
                                        spotlight_df = spotlight_df.drop('Mean_Numeric', axis=1)
                                        
                                        # Reset index
                                        spotlight_df = spotlight_df.reset_index(drop=True)
                                        
                                        # Highlight the selected return
                                        def highlight_selected_asset(row):
                                            if row['Asset'] == selected_return:
                                                return ['background-color: #fff3cd; font-weight: bold'] * len(row)
                                            return [''] * len(row)
                                        
                                        styled_spotlight_df = spotlight_df.style.apply(highlight_selected_asset, axis=1)
                                        
                                        st.dataframe(
                                            styled_spotlight_df,
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                    else:
                                        st.info("No valid data found for spotlight analysis.")
                                else:
                                    st.info(f"No data found for {primary_country} in the selected timeframe.")
                            else:
                                st.info(f"No valid return columns found for {primary_country}.")
                        else:
                            st.info(f"No data found for {primary_country}.")
        else:
            st.warning("No 'country' column found in the dataset. Please check your data structure.")
            st.info("Expected columns: 'country', return columns (e.g., 'eq_tr', 'housing_tr'), 'year'")
    
    # Dataset Information Section
    st.markdown("---")
    st.markdown("## About the Dataset")
    
    st.markdown("""
    ### Citation
    This dashboard uses data from the paper:
    
    **"The Rate of Return on Everything, 1870–2015"**  
    Òscar Jordà, Katharina Knoll, Dmitry Kuvshinov, Moritz Schularick, and Alan M. Taylor  
    *Quarterly Journal of Economics*, Volume 134, Issue 3, August 2019, Pages 1225–1298  
    
    Available at: https://www.frbsf.org/wp-content/uploads/wp2017-25.pdf
    
    ### Key Results
    
    The paper provides the first systematic evidence on the rates of return on different asset classes across 16 advanced economies over the period 1870–2015. Key findings include:
    
    - **Equity returns** have been consistently higher than returns on safe assets (bonds and bills) over the long run, with an average equity premium of about 5-7% per year.
    - **Housing returns** have been comparable to equity returns in many countries, with real housing returns averaging around 7% per year globally.
    - **Safe assets** (government bonds and bills) have provided lower but more stable returns, averaging around 2-3% per year in real terms.
    - The **risk-return tradeoff** is evident: riskier assets (equities, housing) provide higher average returns but with greater volatility.
    - **Real returns** have been remarkably stable across different historical periods, despite major economic and political disruptions.
    
    ### Return Type Definitions
    
    The dataset includes various return measures for different asset classes:
    
    - **`_tr`** (Total Return): The total return including both capital gains and income (dividends, rent, interest). This is the most comprehensive measure.
    - **`_capgain`** (Capital Gain): The return from price appreciation only, excluding income components.
    - **`_rent`** (Rental Yield): For housing, the rental income component of total return.
    - **`_div`** (Dividend Yield): For equities, the dividend income component of total return.
    - **`_dp`** (Dividend Price Ratio): The ratio of dividends to price, a valuation measure.
    - **`_yd`** (Yield): Generic yield measure, often used for bonds or rental yields.
    - **`_rate`** (Rate): Interest rate or return rate, typically for safe assets like bills.
    - **`_rtn`** (Return): Generic return measure.
    
    Asset class prefixes:
    - **`eq_`**: Equity (stocks)
    - **`housing_`**: Real estate / housing
    - **`bond_`**: Government bonds
    - **`bill_`**: Treasury bills (short-term government debt)
    - **`risky_`**: Aggregate risky assets
    - **`safe_`**: Aggregate safe assets
    
    All returns are expressed as decimal fractions (e.g., 0.15 = 15%).
    """)

if __name__ == "__main__":
    main()
