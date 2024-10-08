import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data from a CSV file
file_path = 'pizza_hut_locations.csv'
pizza_data = pd.read_csv(file_path)

# Extract latitude and longitude for clustering, and remove rows with NaN values
coordinates = pizza_data[['latitude', 'longitude']].dropna()

# Apply K-means clustering with 5 clusters as an initial guess
kmeans = KMeans(n_clusters=5, random_state=0).fit(coordinates)
pizza_data = pizza_data.dropna(subset=['latitude', 'longitude'])
pizza_data['cluster'] = kmeans.labels_

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(pizza_data['longitude'], pizza_data['latitude'], c=pizza_data['cluster'], cmap='viridis', marker='o')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='red', s=100, alpha=0.75)  # Plot cluster centers
plt.title('Cluster Distribution of Pizza Hut Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Print the coordinates of the cluster centers
print(centers)
