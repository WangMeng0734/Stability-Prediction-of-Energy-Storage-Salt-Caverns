# Import necessary libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import HOA as HOA

# Load data from an Excel file
file_path = "your file"  # Path to your file
data = pd.read_excel(file_path, index_col=False)  # Read the Excel file into a DataFrame

# Divide the data into input features (X) and output labels (y)
X = data.iloc[:, :15]  # Select the first 15 columns as input features
y = data.iloc[:, -1]  # Select the last column as the output label

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#==================================================================Define the fitness function===================================================================================

# HOA Optimization for LGBM Classifier
def fun_lgbm(params):
    """
    This function defines the fitness function for LGBM.
    It takes hyperparameters as input and returns the average RMSE score using K-fold cross-validation.
    """
    n_estimators, learning_rate, max_depth, num_leaves = params
    kf = KFold(n_splits=5)
    rmse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]

        model = lgb.LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            num_leaves=int(num_leaves)
        )

        model.fit(X_train_kf, y_train_kf)
        y_pred = model.predict(X_val_kf)

        # Calculate RMSE for each fold and store it
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_kf, y_pred)))

    # Return the mean RMSE score
    return np.mean(rmse_scores)

# HOA Optimization for XGBoost Classifier
def fun_xgb(params):
    """
    This function defines the fitness function for XGBoost.
    It takes hyperparameters as input and returns the average RMSE score using K-fold cross-validation.
    """
    n_estimators, learning_rate, max_depth, colsample_bytree = params
    kf = KFold(n_splits=5)
    rmse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]

        model = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            gamma=0,
            colsample_bytree=colsample_bytree
        )

        model.fit(X_train_kf, y_train_kf)
        y_pred = model.predict(X_val_kf)

        # Calculate RMSE for each fold and store it
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_kf, y_pred)))

    # Return the mean RMSE score
    return np.mean(rmse_scores)

# HOA Optimization for RF Classifier
def fun_rf(params):
    """
    This function defines the fitness function for Random Forest.
    It takes hyperparameters as input and returns the average RMSE score using K-fold cross-validation.
    """
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params

    # K-fold cross-validation
    kf = KFold(n_splits=5)
    rmse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]

        # Initialize and train the Random Forest model with given hyperparameters
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            random_state=42
        )

        model.fit(X_train_kf, y_train_kf)
        y_pred = model.predict(X_val_kf)

        # Calculate RMSE for each fold and store it
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_kf, y_pred)))

    # Return the mean RMSE score
    return np.mean(rmse_scores)

#==================================================================Hyperparameter Optimization===================================================================================

# Initialize counters for unique identifiers for each model
lgbm_counter = 0
xgb_counter = 0
rf_counter = 0

# Function to generate a unique identifier for LGBM models
def generate_identifier_lgb(prefix='lgbm'):
    global lgbm_counter
    lgbm_counter += 1
    return f"{prefix}{lgbm_counter}"

# LGBM optimization function using RMSE
def optimize_lgbm():
    """
    This function performs optimization for LightGBM using HOA and returns the best hyperparameters and RMSE.
    """
    N, Dim = 120, 4  # Number of particles and dimensions
    T = 200  # Number of iterations
    LB_lgbm, UB_lgbm = [5, 0.01, 2, 2], [200, 1, 10, 10]  # Hyperparameter bounds

    identifier = generate_identifier_lgb()

    # Optimize using HOA and minimize RMSE
    Best_RMSE_lgbm, Best_Params_lgbm, conv_lgbm = HOA.HOA(
        N, T, LB_lgbm, UB_lgbm, Dim,
        lambda params: fun_lgbm(params),
        identifier=identifier
    )

    # Plot convergence
    plt.plot(conv_lgbm)
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("LGBM Hyperparameter Optimization - RMSE Convergence")
    plt.grid(True)
    plt.show()

    return Best_Params_lgbm, Best_RMSE_lgbm

# Function to generate a unique identifier for XGBoost models
def generate_identifier_xgb(prefix='xgb'):
    global xgb_counter
    xgb_counter += 1
    return f"{prefix}{xgb_counter}"

# XGBoost optimization function using RMSE
def optimize_xgb():
    """
    This function performs optimization for XGBoost using HOA and returns the best hyperparameters and RMSE.
    """
    N, Dim = 120, 4  # Number of particles and dimensions
    T = 200  # Number of iterations
    LB_xgb, UB_xgb = [5, 0.01, 2, 0.5], [200, 1, 10, 1]  # Hyperparameter bounds

    identifier = generate_identifier_xgb()

    # Optimize using HOA and minimize RMSE
    Best_RMSE_xgb, Best_Params_xgb, conv_xgb = HOA.HOA(
        N, T, LB_xgb, UB_xgb, Dim,
        lambda params: fun_xgb(params),
        identifier=identifier
    )

    # Plot convergence
    plt.plot(conv_xgb)
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("XGBoost Hyperparameter Optimization - RMSE Convergence")
    plt.grid(True)
    plt.show()

    return Best_Params_xgb, Best_RMSE_xgb

# Function to generate a unique identifier for Random Forest models
def generate_identifier_rf(prefix='rf'):
    global rf_counter
    rf_counter += 1
    return f"{prefix}{rf_counter}"

# Random Forest optimization function using RMSE
def optimize_rf():
    """
    This function performs optimization for Random Forest using HOA and returns the best hyperparameters and RMSE.
    """
    N, Dim = 120, 4  # Number of particles and dimensions
    T = 200  # Number of iterations
    LB_rf, UB_rf = [10, 1, 2, 1], [200, 10, 10, 10]  # Hyperparameter bounds

    identifier = generate_identifier_rf()

    # Optimize using HOA and minimize RMSE for Random Forest
    Best_RMSE_rf, Best_Params_rf, conv_rf = HOA.HOA(
        N, T, LB_rf, UB_rf, Dim,
        lambda params: fun_rf(params),
        identifier=identifier
    )

    # Plot convergence
    plt.plot(conv_rf)
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("Random Forest Hyperparameter Optimization - RMSE Convergence")
    plt.grid(True)
    plt.show()

    return Best_Params_rf, Best_RMSE_rf

#============================================================Create and train three hybrid base learners==============================================================

# Optimize LGBM model and extract the best hyperparameters
best_params_lgbm_1 = optimize_lgbm()

# Extract and display the best hyperparameters for LGBM
best_params_lgbm_1_values = best_params_lgbm_1[0]
print(f"Best Hyperparameters: {best_params_lgbm_1_values}")

# Initialize the LGBM model with the best hyperparameters
model_lgbm_1 = lgb.LGBMRegressor(
    n_estimators=int(best_params_lgbm_1_values[0]),
    learning_rate=best_params_lgbm_1_values[1],
    max_depth=int(best_params_lgbm_1_values[2]),
    num_leaves=int(best_params_lgbm_1_values[3])
)

# Fit the model on the training data
model_lgbm_1.fit(X_train, y_train)

#============================================================================Create and train three hybrid XGB base learners==============================================================

# Get the best parameters from the XGBoost optimization
best_params_xgb_1 = optimize_xgb()

# Extract the hyperparameters from the first element of the tuple
best_params_xgb_1_values = best_params_xgb_1[0]  # This is the array containing the hyperparameters

# Initialize the XGBoost model using the extracted values (XGBRegressor for regression task)
model_xgb_1 = xgb.XGBRegressor(
    n_estimators=int(best_params_xgb_1_values[0]),  # Accessing n_estimators
    learning_rate=best_params_xgb_1_values[1],      # Accessing learning_rate
    max_depth=int(best_params_xgb_1_values[2]),     # Accessing max_depth
    colsample_bytree=best_params_xgb_1_values[3]    # Accessing colsample_bytree
)

# Fit the model on the training data
model_xgb_1.fit(X_train, y_train)

#============================================================================Create and train three hybrid rf base learners==============================================================

# Get the best parameters from the Random Forest optimization
best_params_rf = optimize_rf()

# Extract the hyperparameters from the first element of the tuple
best_params_rf_values = best_params_rf[0]  # This is the array containing the hyperparameters

# Initialize the RandomForestRegressor model using the extracted values
model_rf_1 = RandomForestRegressor(
    n_estimators=int(best_params_rf_values[0]),  # Accessing n_estimators
    max_depth=int(best_params_rf_values[1]),     # Accessing max_depth
    min_samples_split=int(best_params_rf_values[2]),  # Accessing min_samples_split
    min_samples_leaf=int(best_params_rf_values[3]),   # Accessing min_samples_leaf
    random_state=42  # Ensure reproducibility
)

# Fit the model on the training data
model_rf_1.fit(X_train, y_train)

#==========================================================================Evaluate each base learner and get prediction probabilities==========================================

# Function to evaluate the model
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    This function evaluates the model performance on both training and testing datasets.
    It calculates various metrics such as MSE, RMSE, MAE, R², and VAF for both training and test sets.
    """
    model.fit(X_train, y_train)  # Fit the model on training data
    
    # Predictions for the training set
    y_train_pred = model.predict(X_train)
    
    # Predictions for the test set
    y_test_pred = model.predict(X_test)
    
    # Calculate evaluation metrics for the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_vaf = (1 - np.var(y_train - y_train_pred) / np.var(y_train)) * 100
    
    # Calculate evaluation metrics for the test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_vaf = (1 - np.var(y_test - y_test_pred) / np.var(y_test)) * 100
    
    # Print evaluation metrics for the training set
    print(f"\n{model_name} - Training Set Evaluation Metrics:")
    print(f"MSE: {train_mse:.6f}")
    print(f"RMSE: {train_rmse:.6f}")
    print(f"MAE: {train_mae:.6f}")
    print(f"R²: {train_r2:.6f}")
    print(f"VAF: {train_vaf:.6f}")
    
    # Print evaluation metrics for the test set
    print(f"\n{model_name} - Test Set Evaluation Metrics:")
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    print(f"R²: {test_r2:.6f}")
    print(f"VAF: {test_vaf:.6f}")
    
    return y_train_pred, y_test_pred

# Evaluating each model and saving predictions
# Assuming you have your models already trained as `model_lgbm_1`, `model_xgb_1`, `model_rf_1`, and data `X_train`, `y_train`, `X_test`, `y_test`
print("\nEvaluating LGBM-1:")
y_train_pred_lgbm_1, y_test_pred_lgbm_1 = evaluate_model(model_lgbm_1, 'LightGBM-1', X_train, y_train, X_test, y_test)

print("\nEvaluating XGBoost-1:")
y_train_pred_xgb_1, y_test_pred_xgb_1 = evaluate_model(model_xgb_1, 'XGBoost-1', X_train, y_train, X_test, y_test)

print("\nEvaluating RF-1:")
y_train_pred_rf_1, y_test_pred_rf_1 = evaluate_model(model_rf_1, 'RF-1', X_train, y_train, X_test, y_test)

# =======================================================================Hybrid stacking Model===========================================================================================

# Evaluate Stacking Model and Get Evaluation Metrics for Regression
def print_evaluation_metrics(y_true, y_pred, model_name):
    """
    This function calculates and prints evaluation metrics for regression, including MSE, RMSE, MAE, R², and VAF.
    """
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    explained_variance = (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - Training/Test Set Evaluation Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAPE: {mape:.6f}")
    print(f"VAF: {explained_variance:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")

# Create Stacking Regressor Model with RandomForest as a base learner
stacking_model = StackingRegressor(
    estimators=[('lgbm_1', model_lgbm_1),
                ('xgb_1', model_xgb_1),
                ('rf', model_rf_1)],  # Add Random Forest Regressor
    final_estimator=LinearRegression()  # Linear Regression as the final estimator
)

# Train Stacking Model
stacking_model.fit(X_train, y_train)

# Predict with Stacking Model
stacking_train_pred = stacking_model.predict(X_train)  # Get predicted values for training set
stacking_test_pred = stacking_model.predict(X_test)    # Get predicted values for test set

# Print Stacking Model Evaluation Metrics
print("\nStacking Model - Training Set Evaluation Metrics:")
print_evaluation_metrics(y_train, stacking_train_pred, "Stacking - Training Set")

print("\nStacking Model - Test Set Evaluation Metrics:")
print_evaluation_metrics(y_test, stacking_test_pred, "Stacking - Test Set")
