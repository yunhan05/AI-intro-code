import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
data = pd.read_csv('Pizza Sales Dataset.csv')
grouped_orders = data.groupby('order_id')['pizza_name'].apply(list).tolist()

# Transaction data encoding
encoder = TransactionEncoder()
encoded_orders = encoder.fit_transform(grouped_orders)
transaction_df = pd.DataFrame(encoded_orders, columns=encoder.columns_)

# Applying Apriori algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)

# Generating association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Display top 5 rules
print(rules.head())
