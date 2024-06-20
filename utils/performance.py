# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce
import matplotlib.pyplot as plt
import pandas as pd


def performance_dataframe(n_estimators, X, y) -> pd.DataFrame:   
    perform_dict = {"test" : [], "train": []}
    
    for i in range(1, n_estimators+1):
        # Spliting data, 30% for training and 70% for testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
        
        # Encoding data
        encoder = ce.OrdinalEncoder(cols=['buying_price', 'maint_cost', 'n_doors', 'n_person', 'lug_boot', 'safety'])
        X_train = encoder.fit_transform(X_train)
        X_test  = encoder.transform(X_test)
        
        # Creating Random Forest model
        random_forest = RandomForestClassifier(n_estimators=i)
        
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

def performance_plot(n_estimators, X, y) -> plt.plot:   
    perform_dict = {"test" : [], "train": []}
    
    for i in range(1, n_estimators+1):
        # Spliting data, 30% for training and 70% for testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
        
        # Encoding data
        encoder = ce.OrdinalEncoder(cols=['buying_price', 'maint_cost', 'n_doors', 'n_person', 'lug_boot', 'safety'])
        X_train = encoder.fit_transform(X_train)
        X_test  = encoder.transform(X_test)
        
        # Creating Random Forest model
        random_forest = RandomForestClassifier(n_estimators=i)
        
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

    plt.grid()
    plt.xlabel("Number of iterations")
    plt.ylabel("Performance")
    plt.show()