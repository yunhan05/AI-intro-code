import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Load demographic data from a CSV file
county_data = pd.read_csv('acs2017_county_data.csv')

# Drop rows with missing values in 'IncomePerCap' and 'Poverty' columns
county_data.dropna(subset=['IncomePerCap', 'Poverty'], inplace=True)

# Convert these columns to float for numerical processing
county_data['IncomePerCap'] = county_data['IncomePerCap'].astype(float)
county_data['Poverty'] = county_data['Poverty'].astype(float)

# Selecting the features for clustering
X = county_data[['IncomePerCap', 'Poverty']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
county_data['cluster'] = kmeans.fit_predict(X_scaled)

# Convert CountyId to string and zero-pad it to ensure proper matching with GeoJSON
county_data['CountyId'] = county_data['CountyId'].astype(str).str.zfill(5)

# Load the GeoJSON file, adjusting the 'GEO_ID' to match 'CountyId'
with open('geojson-counties-fips.json', 'r') as file:
    geojson = json.load(file)
    for feature in geojson['features']:
        # Assuming 'GEO_ID' starts with '0500000US', remove this part for matching
        feature['properties']['GEO_ID'] = feature['properties']['GEO_ID'][9:]

# Plotting the results using Plotly Express
fig = px.choropleth(county_data,
                    locations='CountyId',  # Using 'CountyId' as geographic identifier
                    geojson=geojson,  # Using the loaded and adjusted GeoJSON
                    featureidkey='properties.GEO_ID',  # Referencing the adjusted GEO_ID in the properties
                    color='cluster',  # Coloring by cluster assignment
                    hover_name='County',  # Showing county names on hover
                    hover_data=['IncomePerCap', 'Poverty'],  # Additional data shown on hover
                    color_continuous_scale="Viridis",  # Color scale
                    scope="usa",  # Limit the map scope to USA only
                    labels={'cluster': 'Cluster'},  # Labeling the color scale
                    title='Clustering of US Counties based on Income Per Capita and Poverty')  # Title
fig.show()  # Display the figure