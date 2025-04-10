import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Step 1: Create a simple dataset
data = {
    "Size (sq ft)": [500, 1000, 1500, 2000],
    "Price (in Lakhs)": [25, 50, 75, 100]
}
df = pd.DataFrame(data)

# Step 2: Split the dataset into features (X) and target (y)
X = df[["Size (sq ft)"]]  # Feature
y = df["Price (in Lakhs)"]  # Target

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Start MLflow tracking
mlflow.set_experiment("House Price Prediction")
with mlflow.start_run():
    # Step 5: Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 6: Make predictions
    y_pred = model.predict(X_test)

    # Step 7: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Log parameters, metrics, and model
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print("Model logged in MLflow!")

    # Optional: Visualize the results (not logged in MLflow)
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, model.predict(X), color="red", label="Prediction")
    plt.xlabel("Size (sq ft)")
    plt.ylabel("Price (in Lakhs)")
    plt.legend()
    plt.title("House Size vs Price Prediction")
    plt.show()