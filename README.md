# Visualize_Iris_Data_Relationships
This project demonstrates how to load and analyze the famous Iris dataset using Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. The analysis includes visualizing relationships between features and calculating correlation coefficients.

## Sample Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(data = np.c_[iris['data'],iris['target']],
                  columns = iris['feature_names']+['target'])

# Task 1: Visualizing relationships between sepal and petal features
sns.pairplot(df, vars = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
plt.suptitle('Pairplot of Iris Dataset Features', y=1.02)
plt.show()

# Task 2: Calculating correlation coefficients
correlation_matrix = df.corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
