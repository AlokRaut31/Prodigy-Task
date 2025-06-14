import pandas as pd
import matplotlib.pyplot as plt

# Correct file path (use raw string literal with r'' to avoid backslash issues)
df = pd.read_csv(r'C:\Users\Alok Kumar\Desktop\PYTHON\world\API_SL.TLF.TOTL.IN_DS2_en_csv_v2_85425.csv', skiprows=4)

# Use the latest available year in the dataset (e.g., 2022 or 2021)
latest_year = '2022' if '2022' in df.columns else '2021'
df_latest = df[['Country Name', latest_year]].dropna()

# Get top 10 countries by labor force (or total value in this dataset)
df_top10 = df_latest.sort_values(by=latest_year, ascending=False).head(10)

# Plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(df_top10['Country Name'], df_top10[latest_year] / 1e6, color='cornflowerblue')
plt.title(f'Top 10 Countries by Value in {latest_year}')
plt.xlabel('Country')
plt.ylabel('Value (in millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

