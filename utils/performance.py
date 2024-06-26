# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import category_encoders as ce
import matplotlib.pyplot as plt
import pandas as pd

def object_type(X_set):
    dicty = X_set.dtypes.to_dict()
    cols_to_encode = []
    for key, values in dicty.items():
        if values == "object":
            cols_to_encode.append(key)
    return cols_to_encode

def performance_dataframe(n_estimators, X, y, depth=2) -> pd.DataFrame:   
    perform_dict = {"test" : [], "train": []}
    
    for i in range(1, n_estimators+1):
        # Spliting data, 30% for training and 70% for testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
        
        # Encoding data from only categorical type within X columns.
        cols_to_encode = object_type(X)
        if len(cols_to_encode) > 0:
            encoder = ce.OrdinalEncoder(cols=cols_to_encode)
            X_train = encoder.fit_transform(X_train)
            X_test  = encoder.transform(X_test)
        
        # Creating Random Forest model
        random_forest = RandomForestClassifier(n_estimators=i, max_depth=depth, random_state=42)
        
        # Training model
        random_forest.fit(X_train, y_train)
        
        # Predicting
        y_pred_test = random_forest.predict(X_test)
        y_pred_train = random_forest.predict(X_train)
        
        # Metrics
        a_test = accuracy_score(y_test, y_pred_test)
        a_train = accuracy_score(y_train, y_pred_train)
        
        perform_dict["test"].append(a_test)
        perform_dict["train"].append(a_train)
        
        # Dataframe
        perform_data = pd.DataFrame(perform_dict).reset_index().rename(columns={"index" : "iteration"})

    return perform_data

def performance_plot(n_estimators, X, y, depth=2) -> plt.plot:
    """
        performance_plot function takes input X, y data from a clean dataset and calculate the performance of RandomForestClassifier given a number of estimators.
        Arguments:
            n_estimators: the number of estimators to be included in Random Forest Classifier method
            X: X columns from cleaned dataset, if categorical variables are present, these will be encoded using function object_type()
            y: y column from cleaned dataset, always a numerical variable.
    """   
    perform_dict = {"test" : [], "train": []}
    
    for i in range(1, n_estimators+1):
        # Spliting data, 30% for training and 70% for testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
        
        # Encoding data from only categorical type within X columns.
        cols_to_encode = object_type(X)
        if len(cols_to_encode) > 0:
            encoder = ce.OrdinalEncoder(cols=cols_to_encode)
            X_train = encoder.fit_transform(X_train)
            X_test  = encoder.transform(X_test)
        
        # Creating Random Forest model
        random_forest = RandomForestClassifier(n_estimators=i, max_depth=depth, random_state=42)
        
        # Training model
        random_forest.fit(X_train, y_train)
        
        # Predicting
        y_pred_test = random_forest.predict(X_test)
        y_pred_train = random_forest.predict(X_train)
        
        # Metrics
        a_test = accuracy_score(y_test, y_pred_test)
        a_train = accuracy_score(y_train, y_pred_train)
        
        perform_dict["test"].append(a_test)
        perform_dict["train"].append(a_train)
        
        # Dataframe
        perform_data = pd.DataFrame(perform_dict).reset_index().rename(columns={"index" : "iteration"})
        
    (
        perform_data[["test", "train"]]
        .pipe(
            lambda df: (
                df.test.interpolate(method="linear").plot(color="black", marker="o", alpha=6/9, linestyle="dashed"),
                df.train.interpolate(method="linear").plot(color="red", marker="o", alpha=6/9, linestyle="dashed")
            )
        )
    )
    
    # Plot adjustment.
    plt.grid()
    plt.xlabel("Number of iterations")
    plt.ylabel("Performance")
    plt.show()