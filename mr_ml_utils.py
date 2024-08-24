#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def score_dataset(X, y, model):
    """
    Calculate the mean absolute error (MAE) score for a dataset using cross-validation.

    Parameters:
        X (array-like): The input features of the dataset.
        y (array-like): The target variable of the dataset.
        model: The machine learning model used for prediction.

    Returns:
        float: Mean absolute error score.

    """
    
    # Perform cross-validation and calculate MAE score
    score = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error'
    )
    
    # Convert negative scores back to positive
    mae = (-1 * score)
    
    # Calculate the mean of positive MAE scores
    return mae.mean()



def optuna_RF_Reg(X_train, y_train, n_trials, display=False):
    """
    Perform hyperparameter optimization for Random Forest Regression using Optuna.

    Parameters:
        X_train (array-like): The input features of the training dataset.
        y_train (array-like): The target variable of the training dataset.
        n_trials (int): The number of optimization trials to perform.
        display (bool): Whether to display optimization visualizations (default: False).

    Returns:
        dict: The best hyperparameters found during optimization.

    """
    # Import necessary libraries and modules
    import optuna
    from optuna.visualization import plot_contour
    from optuna.visualization import plot_param_importances

    def objective(trial):
        """
        Objective function for Optuna optimization.
        Defines the hyperparameters to optimize and trains Random Forest Regression models with different hyperparameter configurations.

        Parameters:
            trial: An Optuna trial object.

        Returns:
            float: The score (negative mean absolute error) of the trained model.
        """
        # Define the hyperparameters to optimize using Optuna's suggest methods
        RF_reg_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 60, 180, step=20),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 3, 8),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 7),
            max_features=trial.suggest_int("max_features", 3, len(X_train.columns)+1),
            random_state=0
        )

        # Train a Random Forest Regression model with the current hyperparameter configuration
        RF_reg = RandomForestRegressor(**RF_reg_params).fit(X_train, y_train)
        
        # Calculate the score (negative mean absolute error) using the score_dataset() function
        score = score_dataset(X_train, y_train, RF_reg)
        
        return score

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    RFR_best_params = study.best_params

    # Display optimization visualizations if display is True
    if display == True:
        plot_param_importances(study).show()
        plot_contour(study).show()

    # Print the best hyperparameters found during optimization
    print(f'\nThe best hyperparameters for RF Regressor: {RFR_best_params}')

    # Return the best hyperparameters
    return RFR_best_params


def optuna_XGBoost_Reg(X_train, y_train, n_trials, display=False):
    """
    Perform hyperparameter optimization for Random Forest Regression using Optuna.

    Parameters:
        X_train (array-like): The input features of the training dataset.
        y_train (array-like): The target variable of the training dataset.
        n_trials (int): The number of optimization trials to perform.
        display (bool): Whether to display optimization visualizations (default: False).

    Returns:
        dict: The best hyperparameters found during optimization.

    """
    # Import necessary libraries and modules
    import optuna
    from optuna.visualization import plot_contour
    from optuna.visualization import plot_param_importances

    def objective(trial):
        """
        Objective function for Optuna optimization.
        Defines the hyperparameters to optimize and trains Random Forest Regression models with different hyperparameter configurations.

        Parameters:
            trial: An Optuna trial object.

        Returns:
            float: The score (negative mean absolute error) of the trained model.
        """
        # Define the hyperparameters to optimize using Optuna's suggest methods
        XGBR_reg_params = dict(
                max_depth=trial.suggest_int("max_depth", 4, 12),
                learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                n_estimators = trial.suggest_int('n_estimators', 30, 200),
                min_child_weight = trial.suggest_int('min_child_weight', 3, 10),
                gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                #subsample = trial.suggest_float('subsample', 0.01, 1.0, log=True),
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
                reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                #eval_metric = 'mlogloss',
                #use_label_encoder =  'False',
                random_state=0
        )

        # Train a Random Forest Regression model with the current hyperparameter configuration
        XGBR = XGBRegressor(**XGBR_reg_params).fit(X_train, y_train)
        
        # Calculate the score (negative mean absolute error) using the score_dataset() function
        score = score_dataset(X_train, y_train, XGBR)
        
        return score

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    XGBR_best_params = study.best_params

    # Display optimization visualizations if display is True
    if display == True:
        plot_param_importances(study).show()
        plot_contour(study).show()

    # Print the best hyperparameters found during optimization
    print(f'\nThe best hyperparameters for XGBoost Regressor: {XGBR_best_params}')

    # Return the best hyperparameters
    return XGBR_best_params


def modified_learning_curve(X_train, y_train, X_test, y_test, model, N, CV=5, display=False, figure=True):
    """
    Calculate and plot the modified learning curve for a given model.

    Parameters:
        X_train (array-like): The input features of the training dataset.
        y_train (array-like): The target variable of the training dataset.
        X_test (array-like): The input features of the test dataset.
        y_test (array-like): The target variable of the test dataset.
        model: The machine learning model used for prediction.
        N (int): The number of data points to be used in the learning curve.
        CV (int): The number of cross-validation folds (default: 5).
        display (bool): Whether to display intermediate training results (default: False).
        figure (bool): Whether to plot the learning curve (default: True).

    Returns:
        DataFrame: The learning curve results.

    """

    # Generate an array of training sizes
    train_sizes = np.linspace(0.05, 0.99, N)

    # Create an empty DataFrame to store the learning curve results
    results = pd.DataFrame(columns=['train_size', 'mae_train_mean', 'mae_train_std', 'mae_test_mean', 'mae_test_std'],
                           index=range(0, N))

    # Iterate over the training sizes
    for i, train_size in enumerate(train_sizes):
        mae_train = []
        mae_test = []

        # Repeat the cross-validation process for each training size
        for j in range(CV):
            # Split the training set into a subset based on the current training size
            X_train_tmp, _, y_train_tmp, _ = train_test_split(X_train, y_train, test_size=(1 - train_size))

            # Fit the model on the subset of training data
            model.fit(X_train_tmp, y_train_tmp)

            # Make predictions on the test set
            y_pred_test = model.predict(X_test)
            y_pred_train_tmp = model.predict(X_train_tmp)

            # Evaluate the model using mean absolute error
            mae_train.append(mean_absolute_error(y_train_tmp, y_pred_train_tmp))
            mae_test.append(mean_absolute_error(y_test, y_pred_test))

        # Store the learning curve results in the DataFrame
        results['train_size'][i] = len(X_train_tmp)
        results['mae_train_mean'][i] = np.mean(mae_train)
        results['mae_train_std'][i] = np.std(mae_train)
        results['mae_test_mean'][i] = np.mean(mae_test)
        results['mae_test_std'][i] = np.std(mae_test)

        # Print intermediate results if display is True
        if display:
            print(f'For a training size of {len(X_train_tmp)}, MAE(train) = {np.mean(mae_train):.3f} and MAE(test) = {np.mean(mae_test):.3f}')

    # Convert object columns to numeric type in the DataFrame
    for column in results.columns:
        if results[column].dtype == 'object':
            results[column] = pd.to_numeric(results[column], errors='coerce')

    # Plot the learning curve if figure is True
    if figure:
        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")

        plt.plot(results['train_size'], results['mae_train_mean'], 'o-', color="r", label="Training Score")
        plt.fill_between(results['train_size'], results['mae_train_mean'] + results['mae_train_std'],
                         results['mae_train_mean'] - results['mae_train_std'],
                         alpha=0.15, color='r')

        plt.plot(results['train_size'], results['mae_test_mean'], 'o-', color="g", label="Cross-Validation Score")
        plt.fill_between(results['train_size'], results['mae_test_mean'] + results['mae_test_std'],
                         results['mae_test_mean'] - results['mae_test_std'],
                         alpha=0.15, color='g')

        plt.legend(loc="best")
        plt.show()

    return results

def RF_training_with_display(X, y, test_size=0.15, n_trials=20, display=False):
    """
    Train a Random Forest Regression model with hyperparameter optimization and display intermediate results.

    Parameters:
        X (array-like): The input features of the dataset.
        y (array-like): The target variable of the dataset.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.15).
        n_trials (int): The number of optimization trials to perform (default: 20).
        display (bool): Whether to display intermediate training results (default: False).

    Returns:
        tuple: The best model, final trained model on the training data, and final trained model on the full data.

    """

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create a base Random Forest Regression model
    RFR_base_model = RandomForestRegressor(random_state=0)
    # Calculate the base mean absolute error (MAE) score from the training set
    MAE_base = score_dataset(X_train, y_train, RFR_base_model)

    # Print the base MAE score from the training set
    print(f'The base MAE score from the training set is {MAE_base:.0f}')

    # Perform hyperparameter optimization using Optuna and obtain the best hyperparameters
    RFR_best_params = optuna_RF_Reg(X_train, y_train, n_trials, display=display)

    # Create the best model with the obtained hyperparameters
    best_model = RandomForestRegressor(**RFR_best_params)

    # Train the final model on the full data using the best hyperparameters
    final_trained_model_full_data = RandomForestRegressor(**RFR_best_params).fit(X, y)
    # Train the final model on the training data using the best hyperparameters
    final_trained_model_train_data = RandomForestRegressor(**RFR_best_params).fit(X_train, y_train)

    # Make predictions on the training and test sets using the final trained model on the training data
    y_train_predict = final_trained_model_train_data.predict(X_train)
    y_test_predict = final_trained_model_train_data.predict(X_test)
    # Print the mean absolute error (MAE) of the best Random Forest Regressor on the training and test data
    print(f'\nMAE of the best Random Forest Regressor: Training data {mean_absolute_error(y_train, y_train_predict):.3f}')
    print(f'MAE of the best Random Forest Regressor: Test data {mean_absolute_error(y_test, y_test_predict):.3f}')

    # Calculate and plot the modified learning curve for the best model
    results = modified_learning_curve(X_train, y_train, X_test, y_test, final_trained_model_train_data, 5)

    # Return the best model, final trained model on the training data, and final trained model on the full data
    return best_model, final_trained_model_train_data, final_trained_model_full_data


def XGBoost_training_with_display(X, y, test_size=0.15, n_trials=20, display=False):
    """
    Train a Random Forest Regression model with hyperparameter optimization and display intermediate results.

    Parameters:
        X (array-like): The input features of the dataset.
        y (array-like): The target variable of the dataset.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.15).
        n_trials (int): The number of optimization trials to perform (default: 20).
        display (bool): Whether to display intermediate training results (default: False).

    Returns:
        tuple: The best model, final trained model on the training data, and final trained model on the full data.

    """

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create a base Random Forest Regression model
    XGBR_base_model = XGBRegressor(random_state=0)
    # Calculate the base mean absolute error (MAE) score from the training set
    MAE_base = score_dataset(X_train, y_train, XGBR_base_model)

    # Print the base MAE score from the training set
    print(f'The base MAE score from the training set is {MAE_base:.0f}')

    # Perform hyperparameter optimization using Optuna and obtain the best hyperparameters
    XGBR_best_params = optuna_XGBoost_Reg(X_train, y_train, n_trials, display=display)

    # Create the best model with the obtained hyperparameters
    best_model = XGBRegressor(**XGBR_best_params)

    # Train the final model on the full data using the best hyperparameters
    final_trained_model_full_data = XGBRegressor(**XGBR_best_params).fit(X, y)
    # Train the final model on the training data using the best hyperparameters
    final_trained_model_train_data = XGBRegressor(**XGBR_best_params).fit(X_train, y_train)

    # Make predictions on the training and test sets using the final trained model on the training data
    y_train_predict = final_trained_model_train_data.predict(X_train)
    y_test_predict = final_trained_model_train_data.predict(X_test)
    # Print the mean absolute error (MAE) of the best XGBoost Regressor on the training and test data
    print(f'\nMAE of the best XGBoost Regressor: Training data {mean_absolute_error(y_train, y_train_predict):.3f}')
    print(f'MAE of the best XGBoost Regressor: Test data {mean_absolute_error(y_test, y_test_predict):.3f}')

    # Calculate and plot the modified learning curve for the best model
    results = modified_learning_curve(X_train, y_train, X_test, y_test, final_trained_model_train_data, 5)

    # Return the best model, final trained model on the training data, and final trained model on the full data
    return best_model, final_trained_model_train_data, final_trained_model_full_data



def RF_training(X, y, test_size=0.15, n_trials=20):
    """
    Train a Random Forest Regression model with hyperparameter optimization.

    Parameters:
        X (array-like): The input features of the dataset.
        y (array-like): The target variable of the dataset.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.15).
        n_trials (int): The number of optimization trials to perform (default: 20).

    Returns:
        tuple: The best model, final trained model on the training data, and final trained model on the full data.

    """

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create a base Random Forest Regression model
    RFR_base_model = RandomForestRegressor(random_state=0)
    # Calculate the base mean absolute error (MAE) score from the training set
    MAE_base = score_dataset(X_train, y_train, RFR_base_model)

    # Print the base MAE score from the training set
    print(f'The base MAE score from the training set is {MAE_base:.0f}')

    # Perform hyperparameter optimization using Optuna and obtain the best hyperparameters
    RFR_best_params = optuna_RF_Reg(X_train, y_train, n_trials, display=False)

    # Create the best model with the obtained hyperparameters
    best_model = RandomForestRegressor(**RFR_best_params)

    # Train the final model on the full data using the best hyperparameters
    final_trained_model_full_data = RandomForestRegressor(**RFR_best_params).fit(X, y)
    # Train the final model on the training data using the best hyperparameters
    final_trained_model_train_data = RandomForestRegressor(**RFR_best_params).fit(X_train, y_train)

    # Make predictions on the training and test sets using the final trained model on the training data
    y_train_predict = final_trained_model_train_data.predict(X_train)
    y_test_predict = final_trained_model_train_data.predict(X_test)
    # Print the mean absolute error (MAE) of the best Random Forest Regressor on the training and test data
    print(f'\nMAE of the best Random Forest Regressor: Training data {mean_absolute_error(y_train, y_train_predict):.3f}')
    print(f'MAE of the best Random Forest Regressor: Test data {mean_absolute_error(y_test, y_test_predict):.3f}')

    # Return the best model, final trained model on the training data, and final trained model on the full data
    return best_model, final_trained_model_train_data, final_trained_model_full_data

def XGBoost_training(X, y, test_size=0.15, n_trials=20):
    """
    Train a Random Forest Regression model with hyperparameter optimization.

    Parameters:
        X (array-like): The input features of the dataset.
        y (array-like): The target variable of the dataset.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.15).
        n_trials (int): The number of optimization trials to perform (default: 20).

    Returns:
        tuple: The best model, final trained model on the training data, and final trained model on the full data.

    """

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create a base XGBoost Regression model
    XGBR_base_model = XGBRegressor(random_state=0)
    # Calculate the base mean absolute error (MAE) score from the training set
    MAE_base = score_dataset(X_train, y_train, XGBR_base_model)

    # Print the base MAE score from the training set
    print(f'The base MAE score from the training set is {MAE_base:.0f}')

    # Perform hyperparameter optimization using Optuna and obtain the best hyperparameters
    XGBR_best_params = optuna_XGBoost_Reg(X_train, y_train, n_trials, display=False)

    # Create the best model with the obtained hyperparameters
    best_model = XGBRegressor(**XGBR_best_params)

    # Train the final model on the full data using the best hyperparameters
    final_trained_model_full_data = XGBRegressor(**XGBR_best_params).fit(X, y)
    # Train the final model on the training data using the best hyperparameters
    final_trained_model_train_data = XGBRegressor(**XGBR_best_params).fit(X_train, y_train)

    # Make predictions on the training and test sets using the final trained model on the training data
    y_train_predict = final_trained_model_train_data.predict(X_train)
    y_test_predict = final_trained_model_train_data.predict(X_test)
    # Print the mean absolute error (MAE) of the best XGBoost Regressor on the training and test data
    print(f'\nMAE of the best XGBoost Regressor: Training data {mean_absolute_error(y_train, y_train_predict):.3f}')
    print(f'MAE of the best XGBoost Regressor: Test data {mean_absolute_error(y_test, y_test_predict):.3f}')

    # Return the best model, final trained model on the training data, and final trained model on the full data
    return best_model, final_trained_model_train_data, final_trained_model_full_data


