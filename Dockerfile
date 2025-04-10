# Start from a Base Image (ubuntu)
FROM ubuntu:22.04

# Set /usr/app as the default folder
WORKDIR /usr/app

# Copy the code 
COPY . /usr/app

# Define Run ID as an environmental variable
ENV RUN_ID=${RUN_ID}

# Install all the tools and packages
RUN apt update && apt install -y python3 python3-pip python3-venv
 
# Create and Activate Python virtual env
RUN python3 -m venv .venv && . .venv/bin/activate && python3 -m pip install numpy pandas matplotlib scikit-learn mlflow

# Train, Track the Model, Create an experiment 
RUN .venv/bin/activate && python3 app/simple_ml_mlflow.py

# Now set the default command to deploy your model and serve it on port 5000
CMD . .venv/bin/activate && mlflow models serve -m runs:/${RUN_ID}/linear_regression_model 