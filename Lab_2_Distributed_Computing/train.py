from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib

# Coumn Datatypes
column_types = {
    'Id': 'int64',
    'SepalLengthCm': 'float64',
    'SepalWidthCm': 'float64',
    'PetalLengthCm': 'float64',
    'PetalWidthCm': 'float64',
    'Species': 'str'
}

# Iris dataset from Datalake
iris = pd.read_csv('/mnt/datalake/epsilon/iris.csv', sep=',', dtype=column_types)

# Define columns
X = iris.drop(['SepalLengthCm', 'Id', 'Species'], axis=1)  # everything, but ID Species and SepalLengthCM
y = iris['SepalLengthCm']  # Target is SepalLengthCm

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values for the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Save to datalake
joblib.dump(model, '/mnt/datalake/epsilon/iris_linear_model.pkl')
