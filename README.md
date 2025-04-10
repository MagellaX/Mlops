# Setup and Run a simple machine learning program
### Setup
```
git clone https://github.com/basildevops/mlops.git
cd mlops
sudo apt install python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib scikit-learn mlflow
```

### Run the Program without MLFLOW
```
python3 app/simple_ml.py
```
---


# Setup and Run a simple machine learning program with MLFLOW Tracking

Run all the commands under "Setup" section above.

```
python3 -m pip install mlflow
python3 app/simple_ml_mlflow.py  # Program with MLFLOW tracking
ls
```

# Build Your Model into a Container

1. Install Docker using the instructions at https://docs.docker.com/engine/install/ubuntu/

2. Add the current user to docker group

```sudo adduser $USER docker && exit```

5. Build the Container Image. Replace RUN ID with your own.

```mlflow models build-docker -m runs:/1556717bcbab4300b40e328aef0c88ec/linear_regression_model -n linear_regression_model:latest```

# Detailed Break Down
---

### **Step 1: Create a Dataset**
```python
data = {
    "Size (sq ft)": [500, 1000, 1500, 2000, 2500],
    "Price (in Lakhs)": [25, 50, 75, 100, 125]
}
df = pd.DataFrame(data)
```
- **What it does**: 
  - Creates a dictionary with two keys: `"Size (sq ft)"` and `"Price (in Lakhs)"`.
  - Converts this dictionary into a **DataFrame** (a tabular structure) using `pandas`. 
  - The DataFrame looks like this:

| Size (sq ft) | Price (in Lakhs) |
|--------------|------------------|
| 500          | 25               |
| 1000         | 50               |
| 1500         | 75               |
| 2000         | 100              |
| 2500         | 125              |

---

### **Step 2: Separate Features (X) and Target (y)**
```python
X = df[["Size (sq ft)"]]  # Feature
y = df["Price (in Lakhs)"]  # Target
```
- **What it does**:
  - `X` (Feature): Contains the size of the houses, which the model will use as input.
  - `y` (Target): Contains the prices, which the model will predict.

---

### **Step 3: Split the Dataset into Training and Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **What it does**:
  - Splits the data into two parts:
    - **Training set (80%)**: Used to train the model.
    - **Testing set (20%)**: Used to test the model's accuracy.
  - `random_state=42` ensures reproducibility (so the split is always the same).

---

### **Step 4: Train the Linear Regression Model**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
- **What it does**:
  - Creates an instance of the `LinearRegression` model.
  - Trains the model using the training data (`X_train`, `y_train`).
  - During training, the model learns the relationship between house size and price (the "line of best fit").

---

### **Step 5: Make Predictions**
```python
y_pred = model.predict(X_test)
```
- **What it does**:
  - Uses the trained model to predict the house prices for the test data (`X_test`).

---

### **Step 6: Evaluate the Model**
```python
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
```
- **What it does**:
  - Calculates the **Mean Squared Error (MSE)**, which measures the average error in predictions (lower is better).
  - Prints the **model coefficient** (slope of the line) and **intercept** (where the line crosses the y-axis).
  - These help describe the equation of the line: 
    \[
    \text{Price (in Lakhs)} = (\text{Size (sq ft)} \times \text{Coefficient}) + \text{Intercept}
    \]

---

### **Step 7: Visualize the Results**
```python
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Prediction")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price (in Lakhs)")
plt.legend()
plt.title("House Size vs Price Prediction")
plt.show()
```
- **What it does**:
  - Creates a scatter plot of the actual data (blue points).
  - Plots the model's predictions as a red line.
  - Adds labels, a legend, and a title for clarity.
  - Shows the relationship between house size and price.

---

### **How It Works**
1. The program uses **Linear Regression**, which tries to fit a straight line to the data.
2. The slope of the line represents how much the price increases with each additional square foot of size.
3. The intercept represents the predicted price when the size is zero (not realistic but necessary for the model equation).

---

### Example Output
- **Mean Squared Error**: If the error is low (close to zero), the model is accurate.
- **Model Equation**: For example, if the coefficient is `0.05` and the intercept is `0`, the equation is:
  \[
  \text{Price} = 0.05 \times \text{Size}
  \]

---

### What to Try Next
- Add more data points to make the model more realistic.
- Test with other types of models (e.g., Decision Trees).
- Explore multi-variable regression by adding more features like the number of rooms, location, etc.
