import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X):

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        return self
    
    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1.0
        X_scaled = (X - self.min_) / data_range
        
        scale_min, scale_max = self.feature_range
        if scale_min != 0 or scale_max != 1:
            X_scaled = X_scaled * (scale_max - scale_min) + scale_min
        
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self):
        self.categories_ = None
    
    def fit(self, X):
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        for col_idx in range(X.shape[1]):
            unique_categories = np.unique(X[:, col_idx])
            self.categories_.append(unique_categories)
        
        return self
    
    def transform(self, X):
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform. Call fit() first.")
        
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        encoded_columns = []
        
        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            n_categories = len(categories)
            col_encoded = np.zeros((X.shape[0], n_categories), dtype=np.float64)
            
            for sample_idx in range(X.shape[0]):
                value = X[sample_idx, col_idx]
                category_idx = np.where(categories == value)[0]
                
                if len(category_idx) > 0:
                    col_encoded[sample_idx, category_idx[0]] = 1.0
            
            encoded_columns.append(col_encoded)
        X_encoded = np.hstack(encoded_columns)
        
        return X_encoded
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)