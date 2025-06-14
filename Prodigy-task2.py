# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv(r"C:\Users\Alok Kumar\Desktop\PYTHON\titanic\train.csv")  # Update the path to your CSV

# Step 1: Data Cleaning

# Check for missing values
missing_data = df.isnull().sum()
print("Missing Data:\n", missing_data)

# Visualizing missing data
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows where 'Cabin' is missing (or you could try filling it, but we drop it here for simplicity)
df.drop(columns=['Cabin'], inplace=True)

# Step 2: Exploratory Data Analysis (EDA)

# Checking basic statistics of the dataset
print(df.describe())

# Checking the data types of the columns
print(df.dtypes)

# Step 3: Data Visualizations

# 1. Distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Distribution of Pclass
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', palette='coolwarm')
plt.title('Distribution of Passengers by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# 3. Survival Rate by Pclass
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', hue='Survived', palette='coolwarm')
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# 4. Age vs Fare (Scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', palette='coolwarm', alpha=0.6)
plt.title('Age vs Fare by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# 5. Survival Rate by Gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sex', hue='Survived', palette='coolwarm')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Step 4: Correlation Analysis
correlation_matrix = df.corr()

# Plot the heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 5: Feature Engineering (Optional)
# Convert 'Sex' to numeric (binary encoding: male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numeric using label encoding
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 6: Final Cleaned Dataset
print("Cleaned Dataset:\n", df.head())

# Save the cleaned dataset for future use
df.to_csv(r'C:\Users\Alok Kumar\Desktop\Titanic\cleaned_titanic.csv', index=False)
print("Cleaned data saved to 'cleaned_titanic.csv'")
