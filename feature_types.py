from matplotlib import font_manager
from prince import FAMD, PCA, MCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to set up fonts to LaTeX style
def setup_fonts():
    font_dirs = ['./fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
        prop = font_manager.FontProperties(fname=font_file)
        print(font_file, prop.get_name())
    plt.rcParams['font.family'] = 'CMU Serif'

# Function to choose the appropriate model (FAMD, PCA, or MCA) based on the dataset structure
def choose_model(data):
    n_columns = len(data.columns)  # Set n_components to the total number of features
    model = None
    try:
        # Attempt to apply FAMD for mixed data
        model = FAMD(n_components=n_columns, random_state=42)
        model = model.fit(data)
    except ValueError as e:
        print(e, '', end='')
        # Fallback to PCA if the dataset is purely numerical
        if "PCA" in str(e):
            model = PCA(n_components=n_columns, random_state=42)
            model = model.fit(data)
        # Fallback to MCA if the dataset is purely categorical
        elif "MCA" in str(e):
            model = MCA(n_components=n_columns, random_state=42)
            model = model.fit(data)
        else:
            # Raise an error for any other unexpected issue
            raise ValueError(f"Unexpected error: {str(e)}")
    return model

# Function to get total explained variance or inertia and the method name
def get_total_explained_variance_and_method(model):
    if hasattr(model, 'explained_inertia_'):
        # For FAMD or MCA, use explained inertia
        variance = model.explained_inertia_
        method_name = model.__class__.__name__
    elif hasattr(model, 'percentage_of_variance_'):
        # For PCA, use percentage of variance
        variance = model.percentage_of_variance_
        method_name = model.__class__.__name__
    
    # Total explained variance or inertia is the sum of the variances
    total_explained_variance = np.sum(variance)

    return total_explained_variance, method_name

# Plot total explained variance for all datasets
def plot_total_explained_variance(dataset_labels, explained_variances, colors):
    plt.figure(figsize=(21, 9))
    bars = plt.bar(dataset_labels, explained_variances, color=colors)

    # Add dashed lines at every 20 units on the y-axis
    plt.yticks(np.arange(0, 100 + 1, 20))
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    # Set font size for x-ticks and y-ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.title('Total Explained Variance/Inertia for Each Dataset', fontsize=24)
    plt.xlabel('Dataset Id and Method', fontsize=24)
    plt.ylabel('Total Explained Variance/Inertia (%)', fontsize=24)

    # Annotate the bars with the percentage if it is not 100%
    for bar, variance in zip(bars, explained_variances):
        height = bar.get_height()
        if height <= 99:
            plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}%', 
                     ha='center', va='bottom', fontsize=16, color='black')

    plt.tight_layout()
    plt.savefig('run_prince_multi.png')
    plt.show()

if __name__ == "__main__":

    setup_fonts()

    datasets = {
        'binary': [37, 44, 1462, 1479, 1510],
        'multiclass': [23, 181, 1466, 40691, 40975],
        'multilabel': [41465, 41468, 41470, 41471, 41473]
    }

    total_explained_variances = []
    dataset_labels = []
    colors = []
    methods = []

    # Define color mapping for each category
    color_mapping = {
        'binary': 'red',
        'multiclass': 'green',
        'multilabel': 'blue'
    }

    # Iterate through dataset categories
    for category, dataset_list in datasets.items():
        for ds in dataset_list:

            print(ds, '', end='')
            # Load dataset and drop the target column
            data = pd.read_csv(f"./artifacts/autobalancer_datasets/openml_{ds}.csv").drop(columns=["class"])

            # Choose and fit the model
            model = choose_model(data)
            print(model.__class__.__name__,)

            # # Get the total explained variance or inertia and the method name
            # total_explained_variance, method_name = get_total_explained_variance_and_method(model)

            # # Append the results for plotting
            # total_explained_variances.append(total_explained_variance)
            # dataset_labels.append(f"{ds}\n({method_name})")
            # colors.append(color_mapping[category])

    # Plot total explained variance/inertia for all datasets
    plot_total_explained_variance(dataset_labels, total_explained_variances, colors)
