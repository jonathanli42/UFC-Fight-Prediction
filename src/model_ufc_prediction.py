from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def prediction_model(training_data, comp_fighter_data):
    """
    Predict the outcome of a fight between two fighters using a trained machine learning model.

    Parameters:
    -----------
    model : sklearn.base.BaseEstimator
        The trained machine learning model used to predict the fight outcome.
    comp_fighter_data : pandas.DataFrame
        The processed data for the comparison of two fighters. This DataFrame should have all
        the required features for the model to make a prediction.

    Returns:
    --------
    results : dict
        A dictionary containing:
        - 'predicted_winner' (str): The predicted winner ('Fighter A' or 'Fighter B').
        - 'fighter_A_pct_winning' (float): The predicted probability of Fighter A winning.
        - 'fighter_B_pct_winning' (float): The predicted probability of Fighter B winning.
    """

    X = training_data.drop(columns=['target'])
    y = training_data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y)

    param_grid = {
        'n_estimators': [300],
        'max_depth': [3],
        'learning_rate': [0.02],
        'subsample': [0.7],
        'colsample_bytree': [0.9],
        'gamma': [1],
        'min_child_weight': [1],
        'reg_alpha': [0.1],
        'reg_lambda': [2]
    }

    xgb = XGBClassifier(eval_metric='logloss')

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=kfold,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model_xgb2 = grid_search.best_estimator_

    fight_outcome_xgb2 = best_model_xgb2.predict(comp_fighter_data)
    fight_probability_xgb2 = best_model_xgb2.predict_proba(comp_fighter_data)

    prob_A_winning = fight_probability_xgb2[0][0]
    prob_B_winning = fight_probability_xgb2[0][1]

    predicted_winner = 'fighter_1' if fight_outcome_xgb2[0] == 0 else 'fighter_2'

    results = {
        "predicted_winner": predicted_winner,
        "fighter_A_pct_winning": prob_A_winning,
        "fighter_B_pct_winning": prob_B_winning
    }

    return results
