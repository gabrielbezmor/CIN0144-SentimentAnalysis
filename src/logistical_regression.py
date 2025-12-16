import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# Supondo dados já normalizados
def GridSearchLinearRegression(X_train, y_train,param_grid, cv):
    grid_search = GridSearchCV(LogisticRegression(random_state=42),param_grid,cv=cv, scoring="accuracy",n_jobs=1,verbose=1)
    grid_search.fit(X_train, y_train)

    print("Melhores parametros encontrados")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    return best_model

