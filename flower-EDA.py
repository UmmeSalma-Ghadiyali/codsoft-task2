import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suppress FutureWarning related to use_inf_as_na
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
df = pd.read_csv('flower-dataset.csv')

# Convert infinite values to NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display the first few rows of the dataset
print(df.head())

# Get summary statistics of numerical columns
print('\nSummary Statistics:')
print(df.describe())

# Check for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Check the distribution of target classes
print('\nClass Distribution:')
print(df['species'].value_counts())

# Plot histograms for numerical features
plt.figure(figsize=(8, 6))
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle('Pairplot of Features with Species', y=1.02)
plt.show()

# Boxplot for each feature by species
plt.figure(figsize=(8, 6))
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    sns.boxplot(data=df, x='species', y=feature)
    plt.title(f'Boxplot of {feature} by Species')
    plt.xlabel('Species')
    plt.ylabel(feature)
    plt.show()

# Correlation heatmap
correlation_matrix = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()