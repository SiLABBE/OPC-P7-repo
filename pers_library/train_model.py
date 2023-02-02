# Bank Credit Customer
def cv_train(
    name, 
    n_features=20, 
    model_choice='lr', 
    estimators_number=30, 
    apply_SMOTE=True
    ):
    
    import warnings

    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        make_scorer,
        fbeta_score,
    )
    from sklearn.model_selection import (
        train_test_split,
        cross_val_score,
        cross_validate,
        StratifiedKFold,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline

    from imblearn.over_sampling import SMOTE

    import mlflow
    import mlflow.sklearn

    import logging

    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    def eval_metrics(cv):
        cv_time = cv['fit_time'].mean()
        cv_accuracy = cv['test_accuracy'].mean()
        cv_recall = cv['test_recall'].mean()
        cv_f10 = cv['test_f10'].mean()
        cv_auc = cv['test_roc_auc'].mean()
        return cv_time, cv_accuracy, cv_recall, cv_f10, cv_auc

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

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run(run_name=name):
        # Execute LogisticRegression
        if model_choice == 'lr':   
            # Standard scaler trained on train set and applied to both sets
            std = StandardScaler()
            model = LogisticRegression(class_weight='balanced',solver='lbfgs', random_state=42)
            pipeline = make_pipeline(std, model)
        
        elif model_choice == 'rfc':
            model = RandomForestClassifier(n_estimators=estimators_number, class_weight='balanced')
            if apply_SMOTE:
                sm = SMOTE()
                pipeline=make_pipeline(sm, model)
            else:
                pipeline = make_pipeline(model)            

        f10_score = make_scorer(fbeta_score, beta=10)

        skf_5 = StratifiedKFold(5)
        cv_f10 = cross_val_score(pipeline, X_train, y_train, cv=skf_5, scoring=f10_score)
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=skf_5, 
            scoring=['accuracy','recall', 'roc_auc'],
            return_train_score=True
        )
        cv_scores["test_f10"] = cv_f10

        cv_time, cv_accuracy, cv_recall, cv_f10, cv_auc = eval_metrics(cv_scores)

        # Print out metrics
        print("  Fit time: %s" % cv_time)
        print("  Accuracy: %s" % cv_accuracy)
        print("  Recall score: %s" % cv_recall)
        print("  AUC score: %s" % cv_auc)
        print("  F10 score: %s" % cv_f10)

        # Log parameter and metrics to MLflow
        mlflow.log_param("model type", model_choice)
        mlflow.log_param("Features number", n_features)
        mlflow.log_param("Estimators number", estimators_number)
        mlflow.log_param("Apply SMOTE", apply_SMOTE)

        mlflow.log_metric("fit time", cv_time)
        mlflow.log_metric("Accuracy", cv_accuracy)
        mlflow.log_metric("Recall", cv_recall)
        mlflow.log_metric("AUC", cv_auc)
        mlflow.log_metric("F10 score", cv_f10)

        return cv_scores

def train(
    name, 
    n_features=20, 
    model_choice='lr', 
    estimators_number=30, 
    apply_SMOTE=True
    ):
    
    import warnings

    from time import time

    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        fbeta_score,
        confusion_matrix,
        classification_report,
        accuracy_score,
        roc_auc_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline

    from imblearn.over_sampling import SMOTE

    import mlflow
    import mlflow.sklearn

    import logging

    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    def eval_metrics(actual, pred):
        class_report = classification_report(actual, pred)
        confusion_m = confusion_matrix(actual, pred)
        accuracy = accuracy_score(actual, pred)
        recall = recall_score(actual, pred)
        f10 = fbeta_score(actual, pred, beta=10)
        auc = roc_auc_score(actual, pred)
        return class_report, confusion_m, accuracy, recall, f10, auc

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

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run(run_name=name):
        # Execute LogisticRegression
        if model_choice == 'lr':
            # Standard scaler trained on train set and applied to both sets
            std = StandardScaler()
            model = LogisticRegression(class_weight='balanced',solver='lbfgs', random_state=42)
            pipeline = make_pipeline(std, model)
        
        elif model_choice == 'rfc':
            model = RandomForestClassifier(n_estimators=estimators_number, class_weight='balanced')
            if apply_SMOTE:
                sm = SMOTE()
                pipeline=make_pipeline(sm, model)
            else:
                pipeline = make_pipeline(model)

        t0 = time()
        pipeline.fit(X_train, y_train)
        fit_time = time() - t0

        # Evaluate Metrics
        y_pred = pipeline.predict(X_test)
        
        class_report, confusion_m, accuracy, recall, f10, auc = eval_metrics(y_test, y_pred)

        # Print out metrics
        print("  Fit time: %s" % fit_time)
        print("  Accuracy: %s" % accuracy)
        print("  Recall score: %s" % recall)
        print("  AUC score: %s" % auc)
        print("  F10 score: %s" % f10)

        # Log parameter and metrics to MLflow
        mlflow.log_param("model type", model_choice)
        mlflow.log_param("Features number", n_features)
        mlflow.log_param("Estimators number", estimators_number)
        mlflow.log_param("Apply SMOTE", apply_SMOTE)

        mlflow.log_metric("fit time", fit_time)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("F10 score", f10)

        return pipeline, class_report, confusion_m, X_test, y_test