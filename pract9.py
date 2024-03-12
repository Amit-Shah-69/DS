#pract9A

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")
tree_rules = export_text(model, feature_names=iris.feature_names)
print(f"Decision Rules:\n{tree_rules}")

#pract9B

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix, precision_score,recall_score, f1_score
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 10), distortions, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

#pract9C

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
plt.ylabel('Sales')
plt.title('Overall Sales Trend Over Time')
plt.legend()
plt.show()
def plot_category_comparison():
plt.figure(figsize=(8, 6))
for i, category in enumerate(categories):
plt.bar(time_periods + i * 0.2, sales_data[category], width=0.2, label=category)
plt.xlabel('Time Periods')
plt.ylabel('Sales')
plt.title('Comparison of Sales Across Categories')
plt.legend()
plt.show()
def plot_percentage_distribution():
total_sales = np.sum([sales_data[category] for category in categories], axis=0)
labels = [f'{category}: {percentage:.1f}%' for category, percentage in zip(categories, 
total_sales)]
plt.figure(figsize=(8, 8))
plt.pie(total_sales, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Percentage Distribution of Sales Across Categories')
plt.show()
plot_overall_trend()
plot_category_comparison()
plot_percentage_distribution()
