# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Alok Kumar\Desktop\DATA SCIENCE\PYTHON\titanic\train.csv")

# Step 1: Data Cleaning

# Check and display missing data
print("Missing Data:\n", df.isnull().sum())

# Visualizing missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most frequent)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to excessive missing values
df.drop(columns=['Cabin'], inplace=True)

# Step 2: EDA - Basic Info
print("\nDescriptive Statistics:\n", df.describe())
print("\nData Types:\n", df.dtypes)

# Step 3: Data Visualizations

# 1. Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Pclass Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', hue='Pclass', palette='coolwarm', legend=False)
plt.title('Passenger Count by Pclass')
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

# 4. Age vs Fare Scatter Plot
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

# Only numeric columns for correlation matrix
correlation_matrix = df.select_dtypes(include='number').corr()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 5: Feature Engineering

# Encode 'Sex' column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked' column
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 6: Save Cleaned Dataset

print("\nCleaned Dataset Preview:\n", df.head())
save_path = r'C:\Users\Alok Kumar\Desktop\DATA SCIENCE\cleaned_titanic.csv'
df.to_csv(save_path, index=False)
print(f"✅ Cleaned dataset saved to: {save_path}")
