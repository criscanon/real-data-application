import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import folium

def plot_correlation_heatmap(data):
    """
    Plot a correlation matrix heatmap for numeric features in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing numeric features.

    Returns:
        None
    """
    # Select numeric features from the DataFrame
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix
    correlation_matrix = numeric_data.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

def prepare_data(data):
    # Separate features (X) and target variable (y)
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Initialize and train the XGBoost regressor
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2, y_pred

def plot_predicted_vs_actual(y_test, y_pred):
    """
    Plot scatter plot of predicted vs. actual values.

    Args:
        y_test (array-like): Actual values.
        y_pred (array-like): Predicted values.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Predicted vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def create_choropleth_map(data):
    """
    Create a choropleth map showing the count of sales per state.

    Args:
        data (DataFrame): DataFrame containing the data with a 'state' column.

    Returns:
        us_map
    """
    # Count the number of sales per state
    state_counts = data['state'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']

    # Create a base map of the United States centered
    us_map = folium.Map(location=[37.0902, -95.7129], zoom_start=5, min_zoom=5, max_zoom=6)

    # Create a choropleth map with Folium
    choropleth = folium.Choropleth(
        geo_data='https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json',  # GeoJSON of US states
        name='choropleth',
        data=state_counts,
        columns=['State', 'Count'],
        key_on='feature.id',  # Key to join DataFrame with state boundaries
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Sales by State'
    ).add_to(us_map)

    # Add pop-up labels to show the count of sales in each state
    for i, row in state_counts.iterrows():
        folium.Marker(
            location=[0, 0],  # Dummy latitude and longitude, just for the labels to appear in the center of the state
            popup=f"{row['State']}: {row['Count']} sales",
            icon=None
        ).add_to(choropleth.geojson)

    return us_map

def plot_sales_count_over_time(data):
    """
    Plot the sales count over time for new and used boats.

    Args:
        data (DataFrame): DataFrame containing boat sales data.

    Returns:
        None
    """
    # Convert the 'created_date' column to datetime if it's not already
    data['created_date'] = pd.to_datetime(data['created_date'])

    # Filter data for new and used boats
    new_boats = data[data['condition__new'] == 1]
    used_boats = data[data['condition__new'] == 0]

    # Group sales by month for new and used boats separately
    new_sales_by_month = new_boats['created_date'].dt.to_period('M').value_counts().sort_index()
    used_sales_by_month = used_boats['created_date'].dt.to_period('M').value_counts().sort_index()

    # Convert Period index to string for plotting
    new_sales_by_month.index = new_sales_by_month.index.strftime('%Y-%m')
    used_sales_by_month.index = used_sales_by_month.index.strftime('%Y-%m')

    # Create a figure and axes
    plt.figure(figsize=(12, 6))

    # Plot the trend lines with round markers for new and used boats
    plt.plot(new_sales_by_month.index, new_sales_by_month.values, marker='o', color='blue', linestyle='-', linewidth=2, label='New Boats')
    plt.plot(used_sales_by_month.index, used_sales_by_month.values, marker='o', color='green', linestyle='-', linewidth=2, label='Used Boats')

    # Set background color
    plt.gca().set_facecolor('#f7f7f7')

    # Labels and title
    plt.xlabel('Creation Date', fontsize=14)
    plt.ylabel('Number of Sales', fontsize=14)
    plt.title('Sales Count Over Time', fontsize=16)

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()