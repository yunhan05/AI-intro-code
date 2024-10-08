import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
pizza_sales_data = pd.read_csv('Pizza Sales Dataset.csv')

# Prepare the data by selecting relevant features
features = pizza_sales_data[['pizza_category', 'pizza_size', 'unit_price']].copy()  # Use .copy() to explicitly copy the DataFrame slice
target = pizza_sales_data['quantity']

# Convert categorical data to numeric data using LabelEncoder
label_encoder = LabelEncoder()
features.loc[:, 'pizza_category'] = label_encoder.fit_transform(features['pizza_category'])  # Use .loc to ensure modification on the original DataFrame
features.loc[:, 'pizza_size'] = label_encoder.fit_transform(features['pizza_size'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Decision Tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the test data
predictions = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")

# Display some rules from the decision tree
from sklearn.tree import export_text
tree_rules = export_text(model, feature_names=['pizza_category', 'pizza_size', 'unit_price'])
print(tree_rules)
