"""
Module for Analyzing and Modeling Bike Sharing Data.

This module provides classes and methods for analyzing bike sharing data, including functionalities such as data loading, 
exploratory data analysis, time series analysis, user behavior analysis, feature engineering, interaction analysis, 
and regression modeling using decision trees.

Classes:
    - DataAnalyzer: Provides basic data loading and statistical analysis on a dataset.
    - BikeSharingAnalyzer: Offers specialized analysis tools for bike sharing datasets, including correlation analysis, 
      time series analysis, weekly/monthly patterns, user behavior, and interaction analysis.
    - DecisionTreeModel: Trains and evaluates a decision tree regression model, calculates feature importance, and 
      retrieves decision tree structure.

Dependencies:
    - pandas: For data manipulation.
    - matplotlib.pyplot: For data visualization.
    - seaborn: For advanced visualizations.
    - sklearn.tree.DecisionTreeRegressor: For decision tree regression model.
    - sklearn.model_selection.train_test_split: For splitting data into training and testing sets.
    - sklearn.metrics: For model evaluation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class DataAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize DataAnalyzer with the path to the CSV file.

        Parameters:
        - file_path: str, path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load data from the specified CSV file.

        Returns:
        - DataFrame containing the loaded data.
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def get_data_info(self):
        """
        Retrieve and display general information about the dataset.

        Returns:
        - Information about the dataset including column data types and non-null counts.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data.info()

    def get_nan_counts(self):
        """
        Count the number of NaN values in each column.

        Returns:
        - Series with column names as indices and NaN counts as values.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data.isna().sum()

    def get_descriptive_stats(self):
        """
        Retrieve descriptive statistics for the dataset.

        Returns:
        - DataFrame containing descriptive statistics such as mean, median, and standard deviation for numerical columns.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data.describe()


class BikeSharingAnalyzer:
    def __init__(self, data: pd.DataFrame, date_col='dteday', set_index=True):
        """
        Initialize BikeSharingAnalyzer with an existing DataFrame and set a datetime index if specified.

        Parameters:
        - data: DataFrame containing bike sharing data.
        - date_col: str, name of the date column to set as datetime index (default is 'dteday').
        - set_index: bool, whether to set the date_col as the DataFrame index (default is True).
        """
        self.data = data.copy()
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            if set_index:
                self.data.set_index(date_col, inplace=True)
        self.daily_data = self.data.resample('D').sum() if set_index else None

    def calculate_correlation_matrix(self):
        """
        Calculate the correlation matrix for numerical columns in the dataset.

        Returns:
        - DataFrame containing correlation coefficients for each pair of numerical columns.
        """
        return self.data.corr()

    def plot_correlation_matrix(self, figsize=(14, 10)):
        """
        Plot a heatmap of the correlation matrix for the dataset.

        Parameters:
        - figsize: tuple, size of the figure (default is (14, 10)).
        """
        corr_matrix = self.calculate_correlation_matrix()
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_time_series(self, column='cnt', title="Time Series Plot", xlabel="Date", ylabel="Values", resample_rule='D'):
        """
        Plot a time series for the specified column, with optional resampling.

        Parameters:
        - column: str, name of the column to plot (default is 'cnt').
        - title: str, title of the plot.
        - xlabel: str, label for the x-axis.
        - ylabel: str, label for the y-axis.
        - resample_rule: str, resampling rule (default is 'D' for daily).
        """
        data_to_plot = self.daily_data if resample_rule == 'D' else self.data.resample(resample_rule).sum()
        plt.figure(figsize=(14, 6))
        data_to_plot[column].plot()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def plot_resampled_series(self, resample_rule='M', column='cnt', title="Resampled Time Series"):
        """
        Plot resampled data, such as monthly averages, for the specified column.

        Parameters:
        - resample_rule: str, resampling rule (e.g., 'M' for monthly).
        - column: str, name of the column to plot.
        - title: str, title of the plot.
        """
        resampled_data = self.daily_data[column].resample(resample_rule).mean()
        plt.figure(figsize=(14, 6))
        resampled_data.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Average Rentals")
        plt.grid(True)
        plt.show()

    def calculate_weekly_pattern(self, column='cnt'):
        """
        Calculate average values by day of the week for the specified column.

        Parameters:
        - column: str, name of the column to analyze (default is 'cnt').

        Returns:
        - Series with days of the week as indices and average values as values.
        """
        return self.data.groupby(self.data.index.dayofweek)[column].mean()

    def plot_weekly_pattern(self, column='cnt', title="Weekly Patterns"):
        """
        Plot the weekly patterns for the specified column.

        Parameters:
        - column: str, name of the column to plot (default is 'cnt').
        - title: str, title of the plot.
        """
        weekly_pattern = self.calculate_weekly_pattern(column=column)
        plt.figure(figsize=(10, 5))
        weekly_pattern.plot(kind='bar')
        plt.title(title)
        plt.xlabel("Day of the Week")
        plt.ylabel("Average Rentals")
        plt.xticks(ticks=range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=45)
        plt.grid(axis='y')
        plt.show()

    def calculate_monthly_pattern(self, column='cnt'):
        """
        Calculate average values by month for the specified column.

        Parameters:
        - column: str, name of the column to analyze (default is 'cnt').

        Returns:
        - Series with months as indices and average values as values.
        """
        return self.data.groupby(self.data.index.month)[column].mean()

    def plot_monthly_pattern(self, column='cnt', title="Monthly Patterns"):
        """
        Plot the monthly patterns for the specified column.

        Parameters:
        - column: str, name of the column to plot (default is 'cnt').
        - title: str, title of the plot.
        """
        monthly_pattern = self.calculate_monthly_pattern(column=column)
        plt.figure(figsize=(10, 5))
        monthly_pattern.plot(kind='bar')
        plt.title(title)
        plt.xlabel("Month")
        plt.ylabel("Average Rentals")
        plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
        plt.grid(axis='y')
        plt.show()

    def calculate_user_behavior(self, group_column, user_columns):
        """
        Calculate mean rentals grouped by specified column, such as weather conditions.

        Parameters:
        - group_column: str, column by which to group the data (e.g., 'weathersit').
        - user_columns: list, columns to calculate means for.

        Returns:
        - DataFrame containing mean rentals for each user group.
        """
        return self.data.groupby(group_column)[user_columns].mean()

    def plot_user_behavior(self, group_column, user_columns, title="User Behavior Analysis", labels=None):
        """
        Plot user behavior based on specified grouping, such as weather or holiday.

        Parameters:
        - group_column: str, column by which to group the data.
        - user_columns: list, columns to plot.
        - title: str, title of the plot.
        - labels: list, custom labels for x-ticks.
        """
        user_behavior = self.calculate_user_behavior(group_column, user_columns)
        plt.figure(figsize=(10, 6))
        user_behavior.plot(kind='bar', ax=plt.gca())
        plt.title(title)
        plt.xlabel(group_column.capitalize())
        plt.ylabel("Average Rentals")
        if labels:
            plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
        plt.legend(title="User Type")
        plt.grid(axis='y')
        plt.show()

    def create_lagged_features(self, lag_hours=[1, 24], columns=['cnt', 'temp', 'hum', 'windspeed']):
        """
        Create lagged features for specified columns.

        Parameters:
        - lag_hours: list, time steps for lagged features.
        - columns: list, names of columns to create lagged features for.

        Returns:
        - DataFrame with lagged features.
        """
        for col in columns:
            for lag in lag_hours:
                self.data[f'{col}_prev_{lag}_hr'] = self.data[col].shift(lag)
        lagged_data = self.data.dropna()
        return lagged_data

    def plot_lagged_features(self, columns=['cnt'], lags=[1, 24], features=['cnt', 'temp', 'hum'], title_map=None):
        """
        Plot lagged features for given columns and lags.

        Parameters:
            columns: list of str, columns to plot (actual values).
            lags: list of int, list of lag hours to use for lagged features.
            features: list of str, list of main feature names to plot.
            title_map: dict, custom titles for each feature's plot, default is None.
        """
        lagged_data = self.create_lagged_features(columns=columns, lag_hours=lags)
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 5 * n_features), sharex=True)

        if title_map is None:
            title_map = {
                'cnt': "Rental Counts with Lagged Counts",
                'temp': "Temperature Trends with Lagged Temperature",
                'hum': "Humidity Trends with Lagged Humidity"
            }

        for i, feature in enumerate(features):
            axes[i].plot(lagged_data.index, lagged_data[feature], label=f'Current {feature.capitalize()}')
            for lag in lags:
                lagged_column = f'{feature}_prev_{lag}h'
                if lagged_column in lagged_data.columns:
                    axes[i].plot(lagged_data.index, lagged_data[lagged_column], label=f'{lag}-hour Lag {feature.capitalize()}', linestyle='--')

            axes[i].set_title(title_map.get(feature, f"{feature.capitalize()} Trend with Lagged Data"))
            axes[i].set_ylabel(f"{feature.capitalize()} (Normalized)")
            axes[i].legend()
            axes[i].grid(True)

        fig.tight_layout()
        plt.show()

    def calculate_interaction(self, group_columns, target_column='cnt'):
        """
        Calculate mean rentals based on interactions of group columns.

        Parameters:
        - group_columns: list of str, columns to group by.
        - target_column: str, column to calculate means for (default is 'cnt').

        Returns:
        - DataFrame of mean rentals for interactions of the group columns.
        """
        return self.data.groupby(group_columns)[target_column].mean().unstack()

    def plot_interaction(self, group_columns, target_column='cnt', title="Interaction Analysis", labels=None):
        """
        Plot interaction effects of multiple groups on the target column.

        Parameters:
        - group_columns: list of str, columns to group by (e.g., ['season', 'weathersit']).
        - target_column: str, column to plot interactions for.
        - title: str, title of the plot.
        - labels: list, custom labels for x-ticks.
        """
        interaction_data = self.calculate_interaction(group_columns, target_column=target_column)
        plt.figure(figsize=(10, 6))
        interaction_data.plot(kind='bar', ax=plt.gca())
        plt.title(title)
        plt.xlabel(group_columns[0].capitalize())
        plt.ylabel("Average Rentals")
        if labels:
            plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
        plt.legend(title=group_columns[1].capitalize())
        plt.grid(axis='y')
        plt.show()

    def plot_decision_tree(self, model, feature_names, max_depth=3, figsize=(40, 20)):
        """
        Plot the structure of the decision tree model.

        Parameters:
        - model: trained DecisionTreeRegressor model.
        - feature_names: list of str, names of the features used for training.
        - max_depth: int, depth to plot up to (default is 3).
        - figsize: tuple, size of the plot (default is (40, 20)).
        """
        plt.figure(figsize=figsize)
        plot_tree(model, feature_names=feature_names, filled=True, max_depth=max_depth, fontsize=10)
        plt.title("Decision Tree for Prediction")
        plt.show()


class DecisionTreeModel:
    def __init__(self, data: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DecisionTreeModel with dataset and split parameters.

        Parameters:
        - data: DataFrame containing the dataset.
        - features: list of str, feature column names to use for training.
        - target: str, name of the target column.
        - test_size: float, proportion of data to be used for testing (default is 0.2).
        - random_state: int, random state for reproducibility (default is 42).
        """
        self.data = data.dropna(subset=features + [target])  # Drop rows with NaN in selected features/target
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model = DecisionTreeRegressor(random_state=self.random_state)

        self.X = self.data[features]
        self.y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self):
        """
        Train the Decision Tree Regressor on the training dataset.
        """
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Make predictions on the test dataset.

        Returns:
        - Array of predictions for the test set.
        """
        return self.model.predict(self.X_test)

    def evaluate(self):
        """
        Evaluate the model's performance on the test dataset.

        Returns:
        - tuple: Mean Squared Error (MSE) and R² score.
        """
        y_pred = self.predict()
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2

    def get_tree_depth(self):
        """
        Retrieve the depth of the trained decision tree.

        Returns:
        - int: Maximum depth of the decision tree.
        """
        return self.model.tree_.max_depth

    def get_tree_structure(self):
        """
        Retrieve the structure of the decision tree, including number of nodes and leaves.

        Returns:
        - tuple: Number of leaves and nodes in the decision tree.
        """
        n_leaves = self.model.get_n_leaves()
        n_nodes = self.model.tree_.node_count
        return n_leaves, n_nodes

    def feature_importance(self):
        """
        Retrieve the importance of each feature in the decision tree.

        Returns:
        - dict: Mapping of feature names to their importance scores.
        """
        return dict(zip(self.features, self.model.feature_importances_))

    def detailed_node_structure(self):
        """
        Retrieve the detailed structure of each node, including children, features, and thresholds.

        Returns:
        - dict: Detailed structure information of the tree nodes.
        """
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        return {
            "children_left": children_left,
            "children_right": children_right,
            "feature": feature,
            "threshold": threshold,
        }
    
    def return_model(self):
        """
        Return the trained decision tree model instance.

        Returns:
        - DecisionTreeRegressor: The trained model instance.
        """
        return self.model
