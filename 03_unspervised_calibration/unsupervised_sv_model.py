"""
Unsupervised Learning Approach for Stochastic Volatility Models
==============================================================

This implementation creates a discrete stochastic volatility model using unsupervised learning
to generate latent volatility states, avoiding traditional filtering methods.

Model:
y_t = μ_t + exp(h_t/2) * ε_t
h_{t+1} = ξ + φ(h_t - ξ) + η_t

Where:
- y_t: observed returns
- h_t: log-volatility (latent)
- μ_t: mean return
- ξ: long-run log-volatility mean
- φ: persistence parameter
- ε_t ~ N(0,1): return innovation
- η_t ~ N(0,σ²): volatility innovation

Approach:
1. Use clustering/representation learning on returns to identify volatility regimes
2. Generate pseudo-volatility states from unsupervised patterns
3. Construct two likelihood functions: one for returns given volatility, one for volatility dynamics
4. Estimate parameters using joint maximum likelihood without filtering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedSVModel:
    """
    Unsupervised Stochastic Volatility Model Implementation
    """
    
    def __init__(self, n_clusters=5, representation_method='kmeans', window_size=20):
        """
        Initialize the unsupervised SV model
        
        Parameters:
        -----------
        n_clusters : int
            Number of volatility clusters/regimes
        representation_method : str
            Method for unsupervised learning ('kmeans', 'gmm', 'pca', 'ica')
        window_size : int
            Window size for feature extraction
        """
        self.n_clusters = n_clusters
        self.representation_method = representation_method
        self.window_size = window_size
        
        # Model parameters (to be estimated)
        self.mu = None
        self.xi = None
        self.phi = None
        self.sigma_eta = None
        
        # Clustering/representation learning components
        self.cluster_model = None
        self.scaler = StandardScaler()
        
        # Data storage
        self.returns = None
        self.volatility_features = None
        self.volatility_states = None
        self.log_volatility = None
        
    def extract_volatility_features(self, returns):
        """
        Extract features from returns data for volatility clustering
        """
        n_obs = len(returns)
        features = []
        
        for i in range(self.window_size, n_obs):
            # Window of returns
            window = returns[i-self.window_size:i]
            
            # Statistical features
            abs_returns = np.abs(window)
            squared_returns = window**2
            
            feature_vector = [
                np.mean(abs_returns),           # Mean absolute return
                np.std(abs_returns),            # Std of absolute returns
                np.mean(squared_returns),       # Mean squared return (variance proxy)
                np.std(squared_returns),        # Std of squared returns
                np.max(abs_returns),            # Maximum absolute return
                np.min(abs_returns),            # Minimum absolute return
                np.percentile(abs_returns, 75), # 75th percentile
                np.percentile(abs_returns, 25), # 25th percentile
                np.mean(window),                # Mean return
                np.std(window),                 # Return volatility
                np.skew(window) if len(np.unique(window)) > 1 else 0,  # Skewness
                np.kurtosis(window) if len(np.unique(window)) > 1 else 0  # Kurtosis
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit_representation_model(self, features):
        """
        Fit unsupervised learning model to extract volatility patterns
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        if self.representation_method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(features_scaled)
            cluster_centers = self.cluster_model.cluster_centers_
            
        elif self.representation_method == 'gmm':
            self.cluster_model = GaussianMixture(n_components=self.n_clusters, random_state=42)
            cluster_labels = self.cluster_model.fit_predict(features_scaled)
            cluster_centers = self.cluster_model.means_
            
        elif self.representation_method == 'pca':
            # Use PCA for dimensionality reduction then K-means
            pca = PCA(n_components=min(5, features_scaled.shape[1]))
            features_pca = pca.fit_transform(features_scaled)
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(features_pca)
            cluster_centers = self.cluster_model.cluster_centers_
            
        elif self.representation_method == 'ica':
            # Use ICA for source separation then K-means
            ica = FastICA(n_components=min(5, features_scaled.shape[1]), random_state=42)
            features_ica = ica.fit_transform(features_scaled)
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(features_ica)
            cluster_centers = self.cluster_model.cluster_centers_
        
        return cluster_labels, cluster_centers
    
    def generate_volatility_states(self, cluster_labels, features):
        """
        Generate continuous log-volatility states from discrete clusters
        """
        # Calculate cluster-specific volatility levels
        cluster_volatilities = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                # Use mean squared return as volatility proxy for this cluster
                cluster_features = features[cluster_mask]
                mean_var = np.mean(cluster_features[:, 2])  # Mean squared return column
                cluster_vol = np.log(np.sqrt(mean_var + 1e-8))  # Log volatility
                cluster_volatilities.append(cluster_vol)
            else:
                cluster_volatilities.append(0.0)
        
        # Map each observation to its cluster volatility
        volatility_states = np.array([cluster_volatilities[label] for label in cluster_labels])
        
        # Add some smoothing to make states more continuous
        smoothed_states = np.zeros_like(volatility_states)
        alpha = 0.7  # Smoothing parameter
        
        smoothed_states[0] = volatility_states[0]
        for i in range(1, len(volatility_states)):
            smoothed_states[i] = alpha * smoothed_states[i-1] + (1-alpha) * volatility_states[i]
        
        return smoothed_states
    
    def likelihood_returns_given_volatility(self, returns, log_volatility, mu):
        """
        Likelihood function for returns given volatility states
        L1(y_t | h_t, μ) = ∏ N(y_t; μ, exp(h_t))
        """
        residuals = returns - mu
        volatilities = np.exp(log_volatility / 2)
        
        # Avoid zero volatilities
        volatilities = np.maximum(volatilities, 1e-6)
        
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * volatilities**2))
        log_likelihood -= 0.5 * np.sum((residuals / volatilities)**2)
        
        return log_likelihood
    
    def likelihood_volatility_dynamics(self, log_volatility, xi, phi, sigma_eta):
        """
        Likelihood function for volatility dynamics
        L2(h_{t+1} | h_t, ξ, φ, σ_η) = ∏ N(h_{t+1}; ξ + φ(h_t - ξ), σ_η²)
        """
        if len(log_volatility) < 2:
            return -np.inf
        
        h_t = log_volatility[:-1]
        h_t_plus_1 = log_volatility[1:]
        
        predicted_h = xi + phi * (h_t - xi)
        residuals = h_t_plus_1 - predicted_h
        
        log_likelihood = -0.5 * (len(h_t)) * np.log(2 * np.pi * sigma_eta**2)
        log_likelihood -= 0.5 * np.sum(residuals**2) / sigma_eta**2
        
        return log_likelihood
    
    def joint_log_likelihood(self, params, returns, log_volatility):
        """
        Joint log-likelihood function combining both components
        """
        mu, xi, phi, sigma_eta = params
        
        # Parameter constraints
        if sigma_eta <= 0 or abs(phi) >= 1:
            return -np.inf
        
        ll1 = self.likelihood_returns_given_volatility(returns, log_volatility, mu)
        ll2 = self.likelihood_volatility_dynamics(log_volatility, xi, phi, sigma_eta)
        
        return -(ll1 + ll2)  # Negative for minimization
    
    def fit(self, returns):
        """
        Fit the unsupervised SV model
        """
        self.returns = np.array(returns)
        
        print("Step 1: Extracting volatility features...")
        self.volatility_features = self.extract_volatility_features(self.returns)
        
        print("Step 2: Fitting unsupervised representation model...")
        cluster_labels, cluster_centers = self.fit_representation_model(self.volatility_features)
        
        print("Step 3: Generating volatility states...")
        self.log_volatility = self.generate_volatility_states(cluster_labels, self.volatility_features)
        
        # Align returns with volatility states (skip initial window)
        aligned_returns = self.returns[self.window_size:]
        
        print("Step 4: Estimating parameters via joint maximum likelihood...")
        
        # Initial parameter guess
        initial_mu = np.mean(aligned_returns)
        initial_xi = np.mean(self.log_volatility)
        initial_phi = 0.8
        initial_sigma_eta = 0.1
        
        initial_params = [initial_mu, initial_xi, initial_phi, initial_sigma_eta]
        
        # Optimize
        result = optimize.minimize(
            self.joint_log_likelihood,
            initial_params,
            args=(aligned_returns, self.log_volatility),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'disp': True}
        )
        
        # Store estimated parameters
        self.mu, self.xi, self.phi, self.sigma_eta = result.x
        
        print(f"\nEstimated Parameters:")
        print(f"μ (mean return): {self.mu:.6f}")
        print(f"ξ (long-run log-vol): {self.xi:.6f}")
        print(f"φ (persistence): {self.phi:.6f}")
        print(f"σ_η (vol innovation std): {self.sigma_eta:.6f}")
        
        return result
    
    def predict_volatility(self, returns, steps_ahead=1):
        """
        Predict future volatility states
        """
        if self.log_volatility is None:
            raise ValueError("Model must be fitted first")
        
        # Extract features for new returns
        features = self.extract_volatility_features(returns)
        features_scaled = self.scaler.transform(features)
        
        # Predict clusters
        if hasattr(self.cluster_model, 'predict'):
            cluster_labels = self.cluster_model.predict(features_scaled)
        else:
            cluster_labels = self.cluster_model.fit_predict(features_scaled)
        
        # Generate volatility states
        current_log_vol = self.generate_volatility_states(cluster_labels, features)
        
        # Predict ahead using AR(1) model
        predictions = []
        last_h = current_log_vol[-1]
        
        for _ in range(steps_ahead):
            next_h = self.xi + self.phi * (last_h - self.xi)
            predictions.append(next_h)
            last_h = next_h
        
        return np.array(predictions)
    
    def plot_results(self):
        """
        Plot model results and diagnostics
        """
        if self.returns is None:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Returns and estimated volatility
        aligned_returns = self.returns[self.window_size:]
        time_index = range(len(aligned_returns))
        
        ax1 = axes[0, 0]
        ax1.plot(time_index, aligned_returns, label='Returns', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_index, np.exp(self.log_volatility/2), 
                     color='red', label='Estimated Volatility', linewidth=2)
        ax1.set_title('Returns and Estimated Volatility')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Returns', color='blue')
        ax1_twin.set_ylabel('Volatility', color='red')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Log-volatility time series
        ax2 = axes[0, 1]
        ax2.plot(time_index, self.log_volatility, linewidth=2)
        ax2.set_title('Log-Volatility States')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Log-Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volatility clustering visualization
        ax3 = axes[1, 0]
        # Use PCA for 2D visualization of features
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.scaler.transform(self.volatility_features))
        
        cluster_labels, _ = self.fit_representation_model(self.volatility_features)
        scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=cluster_labels, cmap='viridis', alpha=0.6)
        ax3.set_title('Volatility Feature Clustering')
        ax3.set_xlabel('First Principal Component')
        ax3.set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=ax3)
        
        # Plot 4: Residual diagnostics
        ax4 = axes[1, 1]
        if len(self.log_volatility) > 1:
            h_t = self.log_volatility[:-1]
            h_t_plus_1 = self.log_volatility[1:]
            predicted_h = self.xi + self.phi * (h_t - self.xi)
            residuals = h_t_plus_1 - predicted_h
            
            ax4.scatt = ax4.scatter(predicted_h, residuals, alpha=0.6)
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.set_title('Volatility Model Residuals')
            ax4.set_xlabel('Predicted Log-Volatility')
            ax4.set_ylabel('Residuals')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def generate_synthetic_data(n_obs=1000, mu=0.001, xi=-2.0, phi=0.9, sigma_eta=0.1, seed=42):
    """
    Generate synthetic data from the true SV model for testing
    """
    np.random.seed(seed)
    
    # Generate log-volatility process
    h = np.zeros(n_obs)
    h[0] = xi
    
    for t in range(1, n_obs):
        h[t] = xi + phi * (h[t-1] - xi) + sigma_eta * np.random.normal()
    
    # Generate returns
    epsilon = np.random.normal(size=n_obs)
    returns = mu + np.exp(h/2) * epsilon
    
    return returns, h

# Example usage and testing
if __name__ == "__main__":
    print("Unsupervised Stochastic Volatility Model")
    print("="*50)
    
    # Generate synthetic test data
    print("Generating synthetic data...")
    true_params = {'mu': 0.001, 'xi': -2.0, 'phi': 0.9, 'sigma_eta': 0.1}
    synthetic_returns, true_log_vol = generate_synthetic_data(**true_params)
    
    print(f"True parameters: {true_params}")
    
    # Test different representation methods
    methods = ['kmeans', 'gmm', 'pca', 'ica']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing with method: {method}")
        print('='*50)
        
        # Initialize and fit model
        model = UnsupervisedSVModel(
            n_clusters=5, 
            representation_method=method,
            window_size=20
        )
        
        try:
            result = model.fit(synthetic_returns)
            
            print(f"\nParameter Estimation Results ({method}):")
            print(f"μ: True={true_params['mu']:.6f}, Estimated={model.mu:.6f}")
            print(f"ξ: True={true_params['xi']:.6f}, Estimated={model.xi:.6f}")
            print(f"φ: True={true_params['phi']:.6f}, Estimated={model.phi:.6f}")
            print(f"σ_η: True={true_params['sigma_eta']:.6f}, Estimated={model.sigma_eta:.6f}")
            
            # Calculate estimation errors
            errors = {
                'mu': abs(model.mu - true_params['mu']),
                'xi': abs(model.xi - true_params['xi']),
                'phi': abs(model.phi - true_params['phi']),
                'sigma_eta': abs(model.sigma_eta - true_params['sigma_eta'])
            }
            
            print(f"\nAbsolute Errors:")
            for param, error in errors.items():
                print(f"{param}: {error:.6f}")
            
            # Plot results for the best performing method
            if method == 'kmeans':  # Default choice
                model.plot_results()
                
        except Exception as e:
            print(f"Error with method {method}: {e}")
    
    print(f"\n{'='*50}")
    print("Testing complete!")
