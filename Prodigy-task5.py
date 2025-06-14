import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# ---------------- STEP 1: Load Data ----------------
file_path = r'C:\Users\Alok Kumar\Desktop\DATA SCIENCE\PYTHON\traffic_accidents.csv'  # <- Change path accordingly
df = pd.read_csv(file_path)

# ---------------- STEP 2: Clean & Preprocess ----------------
# Convert date and time columns
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
df['Day'] = df['Date'].dt.day_name()

# Drop rows with missing location data
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# ---------------- STEP 3: Time-Based Analysis ----------------
plt.figure(figsize=(10, 5))
sns.countplot(x='Hour', data=df, palette='magma')
plt.title('Accidents by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='Day', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], palette='coolwarm')
plt.title('Accidents by Day of the Week')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# # ---------------- STEP 4: Weather and Road Condition Analysis ----------------
# plt.figure(figsize=(10, 6))
# # df['Weather_Condition'].value_counts().nlargest(10).plot(kind='bar', color='skyblue')
# plt.title('Top 10 Weather Conditions During Accidents')
# plt.xticks(rotation=45)
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
df['Road_Condition'].value_counts().plot(kind='bar', color='orange')
plt.title('Road Conditions During Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- STEP 5: Light Conditions vs Severity ----------------
if 'Light_Condition' in df.columns and 'Severity' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Light_Condition', hue='Severity', data=df, palette='Set2')
    plt.title('Accident Severity by Light Condition')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------- STEP 6: Heatmap of Accident Hotspots ----------------
heatmap_data = df[['Latitude', 'Longitude']].dropna()

map_center = [heatmap_data['Latitude'].mean(), heatmap_data['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=10)

HeatMap(data=heatmap_data.values, radius=8).add_to(m)

# Save and open map
m.save("accident_hotspots_map.html")
print("Heatmap saved as 'accident_hotspots_map.html'")
