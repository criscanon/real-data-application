import matplotlib.pyplot as plt

def plot_numeric_histograms(data):
    """
    Plot histograms for numeric columns in the DataFrame.

    Args:
        data (DataFrame): The input DataFrame containing numeric columns.

    Returns:
        None
    """
    # Select numeric columns only
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Determine the number of rows and columns for subplots
    num_features = len(numeric_data.columns)
    num_cols = 3  # Number of columns for subplots
    num_rows = (num_features - 1) // num_cols + 1  # Number of rows for subplots

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    # Plot histograms for each numeric feature
    for i, column in enumerate(numeric_data.columns):
        ax = axes[i]
        ax.hist(data[column], bins=24, alpha=0.5)
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Hide empty subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_numeric_boxplots(data):
    """
    Plot boxplots for numeric columns in the DataFrame.

    Args:
        data (DataFrame): The input DataFrame containing numeric columns.

    Returns:
        None
    """
    # Select numeric columns only
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Determine the number of rows and columns for subplots
    num_features = len(numeric_data.columns)
    num_cols = 3  # Number of columns for subplots
    num_rows = (num_features - 1) // num_cols + 1  # Number of rows for subplots

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    # Plot boxplots for each numeric feature
    for i, column in enumerate(numeric_data.columns):
        ax = axes[i]
        ax.boxplot(data[column], vert=False)
        ax.set_title(column)
        ax.set_xlabel('Value')

    # Hide empty subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_categorical_barplots(data):
    """
    Plot bar plots for categorical columns in the DataFrame.

    Args:
        data (DataFrame): The input DataFrame containing categorical columns.

    Returns:
        None
    """
    # Find categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Set up plot
    num_plots = len(categorical_columns)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))

    # Flatten axes if only one row
    if num_rows == 1:
        axes = [axes]

    # Iterate over categorical columns and create bar plots
    for i, column in enumerate(categorical_columns):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row][col]

        # Count frequency of each category
        counts = data[column].value_counts()

        # If the variable has more than 10 categories, group less frequent categories into "Others"
        if len(counts) > 12:
            top_categories = counts.head(12)
            other_count = counts.sum() - top_categories.sum()
            top_categories['Others'] = other_count
            top_categories.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        else:
            counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')

        ax.set_title(f'Bar Plot of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()