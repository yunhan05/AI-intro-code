# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pizza_types = pd.read_csv('pizza_types.csv', index_col=None, header=0, encoding='unicode_escape')
pizza_types['category'].value_counts()

orders = pd.read_csv("orders.csv")
orders['date'] = pd.to_datetime(orders['date'])
orders['time'] = pd.to_datetime(orders['time'], format='%H:%M:%S')

order_details = pd.read_csv("order_details.csv")

pizzas = pd.read_csv("pizzas.csv")

All_orders_info = order_details.merge(orders, how='inner', on=['order_id'])

All_pizza_info = pizzas.merge(pizza_types, how="inner", on=['pizza_type_id'])

final = All_orders_info.merge(All_pizza_info, how='inner', on=['pizza_id'])
df = final.sort_values(by=['order_details_id'])

sales_time = df.groupby(pd.Grouper(key='time'))['price'].sum().sort_values(ascending=False)


df['hour'] = pd.to_datetime(df['time']).dt.hour

time = df['hour'].value_counts().sort_values(ascending=False)

import pandas as pd

# Load the dataset
pizza_sales_data = pd.read_csv('Pizza Sales Dataset.csv')

# Calculate total sales and quantity sold for each pizza type
pizza_sales_summary = pizza_sales_data.groupby('pizza_name').agg({
    'quantity': 'sum',  # Sum of quantities sold
    'total_price': 'sum'  # Sum of total sales
}).sort_values(by='total_price', ascending=False)  # Sort by total sales in descending order

# Display the top entries of the sales summary
print(pizza_sales_summary.head())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assume the DataFrame setup and data loading occurs here...

# Set the style for plotting
plt.style.use('ggplot')

# Plot sales over time
time.head(30).to_frame().plot(kind='bar', color='skyblue', figsize=(15, 8))
plt.ylabel('Total Sales', fontsize=12)
plt.title('Peak Time Sales', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Plot the top 10 pizzas by revenue
pizza_id_sales = df.groupby(['pizza_id'])['price'].sum().sort_values(ascending=False)
pizza_id_sales.head(10).to_frame().plot(kind='bar', color='limegreen', figsize=(15, 8))
plt.ylabel('Revenue', fontsize=12)
plt.title('Top 10 Pizzas by Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Plot the top 10 most ordered pizzas
df.groupby('pizza_id').agg('pizza_id').count().sort_values(ascending=False).head(10).to_frame().plot(kind='bar', color='magenta', figsize=(15, 8))
plt.ylabel('Number of Orders', fontsize=12)
plt.title('Top 10 Most Ordered Pizzas', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Plot monthly pizza sales
df['months'] = pd.to_datetime(df['date']).dt.month
df.groupby('months').agg('months').count().sort_values(ascending=False).to_frame().plot(kind='bar', color='limegreen', figsize=(15, 8))
plt.ylabel('Number of Pizzas Ordered', fontsize=12)
plt.title('Monthly Pizza Sales', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Plot the least 10 pizzas by revenue
pizza_id_sales.tail(10).to_frame().plot(kind='bar', color='red', figsize=(15, 8))
plt.ylabel('Revenue', fontsize=12)
plt.title('Least 10 Pizzas by Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Plot the least 10 most ordered pizzas
df.groupby('pizza_id').agg('pizza_id').count().sort_values(ascending=False).tail(10).to_frame().plot(kind='bar', color='darkgray', figsize=(15, 8))
plt.ylabel('Number of Times Ordered', fontsize=12)
plt.title('10 Least Ordered Pizzas', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Save the processed DataFrame to a CSV file
df.to_csv('pizza_place_sales_analysis.csv')

