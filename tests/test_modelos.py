import pytest
from ds_guardian.modelos import evaluar_clasificacion, evaluar_regresion
from ds_guardian.exceptions import ModelAuditingError

def test_evaluar_clasificacion_raises_length_mismatch():
    y_true = [1, 0, 1]
    y_pred = [1, 0]
    with pytest.raises(ModelAuditingError):
        evaluar_clasificacion(y_true, y_pred)

def test_evaluar_regresion_raises_length_mismatch():
    y_true = [1.0, 2.0]
    y_pred = [1.0, 2.0, 3.0]
    with pytest.raises(ModelAuditingError):
        evaluar_regresion(y_true, y_pred)

def test_graficar_importancia_caracteristicas(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from ds_guardian.modelos import graficar_importancia_caracteristicas
    
    X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [4, 3, 2, 1]})
    y = [0, 0, 1, 1]
    clf = RandomForestClassifier(random_state=42).fit(X, y)
    
    plot_file = str(tmp_path / "feat_imp.png")
    df_imp = graficar_importancia_caracteristicas(clf, feature_names=list(X.columns), save_path=plot_file)
    
    assert len(df_imp) == 2
    assert "Feature" in df_imp.columns and "Importance" in df_imp.columns

def test_optimizar_hiperparametros():
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from ds_guardian.modelos import optimizar_hiperparametros
    
    X = pd.DataFrame({'f1': [1, 2, 3, 4, 5, 6], 'f2': [6, 5, 4, 3, 2, 1]})
    y = [0, 0, 0, 1, 1, 1]
    clf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [10, 20], 'max_depth': [2, 4]}
    
    best_model = optimizar_hiperparametros(clf, param_grid=param_grid, X=X, y=y, cv=2, n_iter=2, scoring='accuracy')
    assert best_model is not None
    assert hasattr(best_model, "predict")


