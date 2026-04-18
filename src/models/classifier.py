"""Классификатор для детекции галлюцинаций"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score
import joblib
from typing import Dict, Tuple

class HallucinationClassifier:
    """Классификатор, объединяющий все признаки"""
    
    def __init__(self, n_features: int = 50, use_feature_selection: bool = True, random_seed: int = 42):
        self.selector = SelectKBest(score_func=mutual_info_classif, k=n_features) if use_feature_selection else None
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight='balanced',
            random_state=random_seed
        )
        self.threshold = 0.5
        self.use_feature_selection = use_feature_selection
        self.is_fitted = False
    
    def fit(self, X_y: Dict[str, np.ndarray]):
        """Обучение классификатора"""
        y = X_y["labels"]
        
        # Объединяем все признаки
        X = np.hstack([
            X_y["probe_last_prompt"],
            X_y["internal_scalar_X"],
            X_y["uncertainty_X"],
        ])
        
        # Нормализация
        X = self.scaler.fit_transform(X)
        
        # Отбор признаков
        if self.selector is not None:
            X = self.selector.fit_transform(X, y)
            print(f"Feature selection: {X.shape[1]} features selected")
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.clf, X, y, cv=3, scoring='average_precision')
        print(f"CV PR-AUC: {cv_scores.mean():.4f} (+- {cv_scores.std():.4f})")
        
        # Обучение
        self.clf.fit(X, y)
        self.is_fitted = True
    
    def predict_proba(self, features: Dict[str, np.ndarray]) -> float:
        """Предсказание вероятности галлюцинации"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet")
        
        X = np.hstack([
            features["probe_vec"].reshape(1, -1),
            features["internal_scalars"].reshape(1, -1),
            features["uncertainty"].reshape(1, -1),
        ])
        
        X = self.scaler.transform(X)
        if self.selector is not None:
            X = self.selector.transform(X)
        
        return float(self.clf.predict_proba(X)[0, 1])
    
    def predict(self, features: Dict[str, np.ndarray]) -> Tuple[int, float]:
        """Предсказание с порогом"""
        prob = self.predict_proba(features)
        return int(prob >= self.threshold), prob
    
    def save(self, path: str):
        """Сохранение модели"""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Загрузка модели"""
        return joblib.load(path)