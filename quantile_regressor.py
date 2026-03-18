import numpy as np
from typing import Optional, Tuple

class QuantileRegressor:
    """
    Quantile Regression implementation from scratch using gradient descent.
    
    Quantile regression estimates conditional quantiles of the response variable
    instead of the conditional mean (like OLS regression).
    """
    
    def __init__(self, quantile: float = 0.5, learning_rate: float = 0.01, 
                 n_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Parameters:
        -----------
        quantile : float
            The quantile to estimate (between 0 and 1). Default is 0.5 (median).
        learning_rate : float
            Step size for gradient descent.
        n_iterations : int
            Maximum number of iterations for optimization.
        tolerance : float
            Convergence tolerance for stopping criterion.
        """
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1")
        
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []
    
    def _check_loss(self, residuals: np.ndarray) -> float:
        """
        Compute the quantile loss (also called pinball loss or tilted loss).
        
        Loss = sum(quantile * max(residual, 0) + (1 - quantile) * max(-residual, 0))
        """
        loss = np.where(residuals >= 0,
                       self.quantile * residuals,
                       (self.quantile - 1) * residuals)
        return np.sum(loss)
    
    def _compute_gradient(self, X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """
        Compute gradient of the quantile loss with respect to parameters.
        
        The subgradient is:
        - quantile if residual > 0
        - quantile - 1 if residual < 0
        - between (quantile - 1) and quantile if residual = 0
        """
        # For residuals = 0, we use the quantile value
        grad_residuals = np.where(residuals > 0, self.quantile, self.quantile - 1)
        gradient = -X.T @ grad_residuals
        return gradient / len(residuals)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        Fit the quantile regression model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : QuantileRegressor
        """
        # Convert to numpy arrays and ensure correct shape
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        
        # Initialize parameters randomly
        params = np.random.randn(n_features + 1) * 0.01
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute predictions and residuals
            predictions = X_with_intercept @ params
            residuals = y - predictions
            
            # Compute and store loss
            loss = self._check_loss(residuals)
            self.loss_history_.append(loss)
            
            # Compute gradient
            gradient = self._compute_gradient(X_with_intercept, residuals)
            
            # Update parameters
            params_new = params - self.learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(params_new - params) < self.tolerance:
                params = params_new
                print(f"Converged after {iteration + 1} iterations")
                break
            
            params = params_new
        
        # Store fitted parameters
        self.intercept_ = params[0]
        self.coef_ = params[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the quantile loss on the given test data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            Negative quantile loss (higher is better)
        """
        predictions = self.predict(X)
        residuals = y - predictions
        return -self._check_loss(residuals)


# Example usage and demonstration
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    X = np.linspace(0, 10, n_samples)
    
    # True relationship with heteroscedastic noise
    y_true = 2 + 3 * X
    noise = np.random.normal(0, 1 + 0.5 * X, n_samples)
    y = y_true + noise
    
    # Fit quantile regression for different quantiles
    quantiles = [0.1, 0.5, 0.9]
    models = {}
    
    print("Fitting Quantile Regression models...\n")
    for q in quantiles:
        print(f"Quantile: {q}")
        model = QuantileRegressor(quantile=q, learning_rate=0.01, 
                                  n_iterations=2000, tolerance=1e-6)
        model.fit(X, y)
        models[q] = model
        print(f"Intercept: {model.intercept_:.4f}, Coefficient: {model.coef_[0]:.4f}")
        print(f"Final loss: {model.loss_history_[-1]:.4f}\n")
    
    # Make predictions
    X_test = np.array([2.5, 5.0, 7.5])
    print("\nPredictions for X = [2.5, 5.0, 7.5]:")
    for q in quantiles:
        predictions = models[q].predict(X_test)
        print(f"Quantile {q}: {predictions}")
    
    # Plot results (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='Data')
        
        X_plot = np.linspace(0, 10, 100)
        for q in quantiles:
            y_pred = models[q].predict(X_plot)
            plt.plot(X_plot, y_pred, label=f'Quantile {q}', linewidth=2)
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Quantile Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('quantile_regression.png', dpi=150)
        print("\nPlot saved as 'quantile_regression.png'")
        
    except ImportError:
        print("\nMatplotlib not available for plotting")