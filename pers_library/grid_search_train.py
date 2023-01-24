def grid_search_train( 
    n_features=20,
    model_choice='lr',
    solver_list=['lbfgs'],
    C_list=[1],
    n_estimator_list=[30],
    max_depth_list=[1000],
    SMOTE=False
    ):
    
    import warnings

    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        fbeta_score,
        accuracy_score,
        roc_auc_score,
        recall_score,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    import mlflow
    import mlflow.sklearn

    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    from hyperopt.pyll import scope

    import logging

    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    def eval_metrics(actual, pred):
        # Calculate metrics from the prediction
        accuracy = accuracy_score(actual, pred)
        recall = recall_score(actual, pred)
        f10 = fbeta_score(actual, pred, beta=10)
        auc = roc_auc_score(actual, pred)
        return accuracy, recall, f10, auc

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the customer csv file
    csv_file = (
        'df_train_' + str(n_features) + ".csv"
    )
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV. Error: %s", e
        )

    # The predicted column is "TARGET" which is 0 (good borrower) or 1 (bad borrower)
    x = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y = data['TARGET']

    # Split the data into training and test sets. (0.8, 0.2) split.
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)
    if SMOTE:
        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)

    def objective_lr(params, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
        # MLops tracking for Logistic Regression
        with mlflow.start_run():
            mlflow.set_tag("model", "Logistic_Regression")
            mlflow.log_params(params)
            
            std = StandardScaler().fit(X_train)
            X_train = std.transform(X_train)
            X_test = std.transform(X_test)
            lr = LogisticRegression(**params)
            lr.fit(X_train,y_train)

            y_pred = lr.predict(X_test)

            accuracy, recall, f10, auc = eval_metrics(y_test, y_pred)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("F10 score", f10)
        return {'loss': f10, 'status': STATUS_OK}

    def objective_rfc(params, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
        # MLops tracking for Random Forest classifier
        with mlflow.start_run():
            mlflow.set_tag("model", "Random_Forest_Classifier")
            mlflow.log_params(params)
            
            rfc = RandomForestClassifier(**params)
            rfc.fit(X_train,y_train)

            y_pred = rfc.predict(X_test)

            accuracy, recall, f10, auc = eval_metrics(y_test, y_pred)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("F10 score", f10)
        return {'loss': f10, 'status': STATUS_OK}

    if model_choice=='lr':
            search_space = {
                "solver": hp.choice("solver", solver_list),
                "C": hp.choice("C", C_list),
                "class_weight" : hp.choice("class_weight", ["balanced"])
                } 

            best_result = fmin(
                fn=objective_lr,
                space=search_space,
                algo=tpe.suggest,
                max_evals=10,
                trials=Trials()
                )
    
    if model_choice=='rfc':
            search_space = {
                "n_estimators": hp.choice("C", n_estimator_list),
                "max_depth": hp.choice("max_depth", max_depth_list),
                "class_weight" : hp.choice("class_weight", ["balanced"])
                } 
            
            best_result = fmin(
                fn=objective_rfc,
                space=search_space,
                algo=tpe.suggest,
                max_evals=10,
                trials=Trials()
                )

    return best_result