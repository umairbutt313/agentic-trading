import numpy as np
from scipy.optimize import differential_evolution

class PortfolioConstructor:
    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray):
        self.returns = returns
        self.cov_matrix = cov_matrix
        
    def optimize(self) -> np.ndarray:
        """Quantum-inspired portfolio optimization"""
        n_assets = len(self.returns)
        bounds = [(0, 0.1)] * n_assets
        
        result = differential_evolution(
            self._objective_function,
            bounds=bounds,
            constraints=self._get_constraints(),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        return result.x

    def _objective_function(self, weights: np.ndarray) -> float:
        port_return = np.dot(weights, self.returns)
        port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        return - (port_return / port_vol)  # Maximize Sharpe ratio

    def _get_constraints(self) -> List[Dict]:
        return [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: self._entropy_constraint(w)}
        ]

    def _entropy_constraint(self, weights: np.ndarray) -> float:
        return -np.sum(weights * np.log(weights)) - 5.0