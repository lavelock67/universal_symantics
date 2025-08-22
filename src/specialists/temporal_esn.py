"""Echo State Network (ESN) for temporal processing and memory.

This module implements an Echo State Network specialist for handling temporal
streams and providing cheap, robust memory for the periodic primitives system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs

logger = logging.getLogger(__name__)


class EchoStateBlock:
    """Echo State Network block for temporal processing.
    
    This class implements an ESN as described in the plan: cheap, robust memory
    that aligns with the "difference-through-time" memory principle.
    """
    
    def __init__(self, size: int = 512, spectral_radius: float = 0.9, 
                 sparsity: float = 0.01, input_scaling: float = 1.0,
                 noise_level: float = 0.01, leak_rate: float = 0.1):
        """Initialize the Echo State Network.
        
        Args:
            size: Size of the reservoir (number of neurons)
            spectral_radius: Spectral radius of the reservoir weight matrix
            sparsity: Sparsity of the reservoir connections
            input_scaling: Scaling factor for input weights
            noise_level: Level of noise to add during training
            leak_rate: Leak rate for leaky integrator neurons
        """
        self.size = size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.noise_level = noise_level
        self.leak_rate = leak_rate
        
        # Initialize reservoir state
        self.reservoir_state = np.zeros(size)
        
        # Initialize weight matrices
        self._initialize_weights()
        
        logger.info(f"Initialized ESN with size={size}, spectral_radius={spectral_radius}")
    
    def _initialize_weights(self) -> None:
        """Initialize the weight matrices for the ESN."""
        # Reservoir weight matrix (sparse, random)
        self.W_res = sparse_random(
            self.size, self.size, 
            density=1.0 - self.sparsity, 
            random_state=42
        ).toarray()
        
        # Normalize to achieve desired spectral radius
        eigenvalues = eigs(self.W_res, k=1, return_eigenvectors=False)
        max_eigenvalue = np.abs(eigenvalues[0])
        self.W_res = self.W_res * (self.spectral_radius / max_eigenvalue)
        
        # Input weight matrix (random)
        self.W_in = np.random.randn(self.size, 1) * self.input_scaling
        
        # Output weight matrix (will be trained)
        self.W_out = None
        
        # Bias terms
        self.b_res = np.random.randn(self.size) * 0.1
        self.b_out = np.random.randn(1) * 0.1
    
    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Perform one step of the ESN.
        
        Args:
            x_t: Input at time t (scalar or 1D array)
            
        Returns:
            Reservoir state at time t
        """
        # Ensure input is 2D
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        
        # Update reservoir state (leaky integrator)
        new_state = (1 - self.leak_rate) * self.reservoir_state + \
                   self.leak_rate * np.tanh(
                       self.W_res @ self.reservoir_state + 
                       self.W_in @ x_t + 
                       self.b_res
                   )
        
        # Add noise during training
        if self.noise_level > 0:
            new_state += np.random.normal(0, self.noise_level, self.size)
        
        self.reservoir_state = new_state
        return self.reservoir_state.copy()
    
    def reset_state(self) -> None:
        """Reset the reservoir state to zero."""
        self.reservoir_state = np.zeros(self.size)
    
    def get_state(self) -> np.ndarray:
        """Get the current reservoir state.
        
        Returns:
            Current reservoir state
        """
        return self.reservoir_state.copy()
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, 
              washout: int = 50, ridge_reg: float = 1e-6) -> Dict[str, float]:
        """Train the ESN output weights.
        
        Args:
            inputs: Input sequence (T, 1) or (T,)
            targets: Target sequence (T, 1) or (T,)
            washout: Number of initial steps to discard
            ridge_reg: Ridge regression regularization parameter
            
        Returns:
            Dictionary with training metrics
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        
        T = inputs.shape[0]
        logger.info(f"Training ESN on {T} time steps with washout={washout}")
        
        # Reset reservoir state
        self.reset_state()
        
        # Collect reservoir states
        reservoir_states = []
        
        # Warm up the reservoir
        for t in range(washout):
            self.step(inputs[t])
        
        # Collect states for training
        for t in range(washout, T):
            state = self.step(inputs[t])
            reservoir_states.append(state)
        
        # Prepare training data
        X = np.array(reservoir_states)  # (T-washout, size)
        y = targets[washout:]  # (T-washout, 1)
        
        # Train output weights using ridge regression
        X_augmented = np.column_stack([X, np.ones(X.shape[0])])  # Add bias term
        W_out_augmented = np.linalg.solve(
            X_augmented.T @ X_augmented + ridge_reg * np.eye(X_augmented.shape[1]),
            X_augmented.T @ y
        )
        
        # Extract weights and bias
        self.W_out = W_out_augmented[:-1].reshape(-1, 1)
        self.b_out = W_out_augmented[-1]
        
        # Calculate training error
        predictions = X_augmented @ W_out_augmented
        mse = np.mean((predictions - y) ** 2)
        
        logger.info(f"ESN training complete. MSE: {mse:.6f}")
        
        return {
            "mse": mse,
            "training_steps": T - washout,
            "reservoir_size": self.size,
        }
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions using the trained ESN.
        
        Args:
            inputs: Input sequence (T, 1) or (T,)
            
        Returns:
            Predicted outputs (T, 1)
        """
        if self.W_out is None:
            raise ValueError("ESN must be trained before making predictions")
        
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        
        T = inputs.shape[0]
        predictions = []
        
        # Reset reservoir state
        self.reset_state()
        
        # Generate predictions
        for t in range(T):
            state = self.step(inputs[t])
            prediction = self.W_out.T @ state + self.b_out
            predictions.append(prediction[0])
        
        return np.array(predictions).reshape(-1, 1)
    
    def compute_memory_capacity(self, max_delay: int = 20, n_trials: int = 10) -> float:
        """Compute the memory capacity of the ESN.
        
        Args:
            max_delay: Maximum delay to test
            n_trials: Number of trials for averaging
            
        Returns:
            Memory capacity score
        """
        logger.info(f"Computing memory capacity (max_delay={max_delay}, n_trials={n_trials})")
        
        mc_scores = []
        
        for trial in range(n_trials):
            # Generate random input sequence
            T = 1000
            inputs = np.random.uniform(-1, 1, T)
            
            # Test different delays
            for delay in range(1, max_delay + 1):
                # Create target: input delayed by 'delay' steps
                targets = np.zeros(T)
                targets[delay:] = inputs[:-delay]
                
                # Train ESN
                self.train(inputs.reshape(-1, 1), targets.reshape(-1, 1), washout=100)
                
                # Test prediction
                predictions = self.predict(inputs.reshape(-1, 1))
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(targets[100:], predictions[100:, 0])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                mc_scores.append(correlation ** 2)
        
        # Average memory capacity
        avg_mc = np.mean(mc_scores)
        logger.info(f"Memory capacity: {avg_mc:.4f}")
        
        return avg_mc


class TemporalESNSpecialist:
    """Temporal specialist using Echo State Networks.
    
    This class implements the temporal specialist mentioned in the plan,
    providing temporal integration and memory for the periodic primitives system.
    """
    
    def __init__(self, reservoir_size: int = 512, n_esns: int = 4):
        """Initialize the temporal ESN specialist.
        
        Args:
            reservoir_size: Size of each ESN reservoir
            n_esns: Number of ESNs to use (for different temporal scales)
        """
        self.reservoir_size = reservoir_size
        self.n_esns = n_esns
        
        # Initialize multiple ESNs with different parameters
        self.esns = []
        for i in range(n_esns):
            # Vary spectral radius and leak rate for different temporal scales
            spectral_radius = 0.7 + 0.2 * (i / (n_esns - 1))  # 0.7 to 0.9
            leak_rate = 0.05 + 0.15 * (i / (n_esns - 1))      # 0.05 to 0.2
            
            esn = EchoStateBlock(
                size=reservoir_size,
                spectral_radius=spectral_radius,
                leak_rate=leak_rate,
                sparsity=0.01,
                input_scaling=1.0,
                noise_level=0.01
            )
            self.esns.append(esn)
        
        # Temporal memory buffer
        self.memory_buffer = []
        self.max_buffer_size = 1000
        
        logger.info(f"Initialized TemporalESNSpecialist with {n_esns} ESNs")
    
    def process_sequence(self, sequence: List[Any]) -> Dict[str, np.ndarray]:
        """Process a temporal sequence through all ESNs.
        
        Args:
            sequence: List of inputs over time
            
        Returns:
            Dictionary with ESN outputs and temporal features
        """
        if not sequence:
            return {}
        
        # Convert sequence to numerical format
        numerical_sequence = self._convert_to_numerical(sequence)
        
        # Process through each ESN
        esn_outputs = {}
        temporal_features = []
        
        for i, esn in enumerate(self.esns):
            # Reset ESN state
            esn.reset_state()
            
            # Process sequence
            states = []
            for t, input_t in enumerate(numerical_sequence):
                state = esn.step(input_t)
                states.append(state)
            
            # Store outputs
            esn_outputs[f"esn_{i}"] = np.array(states)
            
            # Extract temporal features
            features = self._extract_temporal_features(states)
            temporal_features.append(features)
        
        # Combine temporal features
        combined_features = np.concatenate(temporal_features, axis=1)
        
        # Update memory buffer
        self._update_memory_buffer(sequence)
        
        return {
            "esn_outputs": esn_outputs,
            "temporal_features": combined_features,
            "sequence_length": len(sequence),
            "memory_buffer_size": len(self.memory_buffer),
        }
    
    def _convert_to_numerical(self, sequence: List[Any]) -> np.ndarray:
        """Convert sequence to numerical format for ESN processing.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Numerical sequence
        """
        numerical = []
        
        for item in sequence:
            if isinstance(item, (int, float)):
                numerical.append(item)
            elif isinstance(item, str):
                # Simple hash-based conversion
                numerical.append(hash(item) % 1000 / 1000.0)
            elif isinstance(item, np.ndarray):
                # Use mean of array
                numerical.append(np.mean(item))
            elif isinstance(item, dict):
                # Use hash of string representation
                numerical.append(hash(str(item)) % 1000 / 1000.0)
            else:
                # Default to 0
                numerical.append(0.0)
        
        return np.array(numerical).reshape(-1, 1)
    
    def _extract_temporal_features(self, states: List[np.ndarray]) -> np.ndarray:
        """Extract temporal features from ESN states.
        
        Args:
            states: List of reservoir states
            
        Returns:
            Temporal features
        """
        if not states:
            return np.array([])
        
        states_array = np.array(states)
        
        # Extract various temporal features
        features = []
        
        # Mean activation
        features.append(np.mean(states_array, axis=0))
        
        # Standard deviation
        features.append(np.std(states_array, axis=0))
        
        # Temporal derivatives (first differences)
        if len(states_array) > 1:
            derivatives = np.diff(states_array, axis=0)
            features.append(np.mean(derivatives, axis=0))
            features.append(np.std(derivatives, axis=0))
        else:
            features.append(np.zeros(states_array.shape[1]))
            features.append(np.zeros(states_array.shape[1]))
        
        # Spectral features (FFT magnitude)
        if len(states_array) > 1:
            fft_magnitudes = np.abs(np.fft.fft(states_array, axis=0))
            features.append(np.mean(fft_magnitudes, axis=0))
        else:
            features.append(np.zeros(states_array.shape[1]))
        
        return np.concatenate(features)
    
    def _update_memory_buffer(self, sequence: List[Any]) -> None:
        """Update the temporal memory buffer.
        
        Args:
            sequence: New sequence to add to memory
        """
        self.memory_buffer.extend(sequence)
        
        # Maintain buffer size
        if len(self.memory_buffer) > self.max_buffer_size:
            self.memory_buffer = self.memory_buffer[-self.max_buffer_size:]
    
    def predict_next(self, context: List[Any], horizon: int = 5) -> List[Any]:
        """Predict next elements in the sequence.
        
        Args:
            context: Context sequence
            horizon: Prediction horizon
            
        Returns:
            List of predicted next elements
        """
        if not context or horizon <= 0:
            return []
        
        # Convert context to numerical
        numerical_context = self._convert_to_numerical(context)
        
        # Use the first ESN for prediction
        esn = self.esns[0]
        
        # Train on context
        targets = numerical_context[1:]  # Predict next element
        inputs = numerical_context[:-1]
        
        if len(inputs) > 1:
            esn.train(inputs, targets, washout=min(10, len(inputs)//4))
            
            # Generate predictions
            predictions = esn.predict(numerical_context[-horizon:])
            
            # Convert back to original format (simplified)
            return [float(p) for p in predictions.flatten()]
        else:
            return [0.0] * horizon
    
    def compute_temporal_coherence(self, sequence: List[Any]) -> float:
        """Compute temporal coherence using normalized autocorrelation.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Coherence score in [0, 1]
        """
        if not sequence or len(sequence) < 3:
            return 0.0
        
        # Convert to float array
        x = np.asarray(sequence, dtype=float)
        x = x - np.mean(x)
        denom = np.sum(x * x) + 1e-10
        
        # Compute lag-1 autocorrelation as a simple coherence proxy
        ac_lag1 = np.sum(x[:-1] * x[1:]) / denom
        
        # Map [-1, 1] to [0, 1]
        coherence = 0.5 * (ac_lag1 + 1.0)
        
        # Guard numerical bounds
        return float(max(0.0, min(1.0, coherence)))
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the temporal memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.memory_buffer:
            return {"buffer_size": 0, "temporal_patterns": []}
        
        # Analyze temporal patterns in memory
        numerical_buffer = self._convert_to_numerical(self.memory_buffer)
        
        # Compute basic statistics
        mean_val = np.mean(numerical_buffer)
        std_val = np.std(numerical_buffer)
        
        # Detect temporal patterns (simplified)
        patterns = []
        if len(numerical_buffer) > 10:
            # Look for repeating patterns
            autocorr = np.correlate(numerical_buffer.flatten(), 
                                  numerical_buffer.flatten(), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            
            if peaks:
                patterns.append(f"Repeating pattern with period {peaks[0]}")
        
        return {
            "buffer_size": len(self.memory_buffer),
            "mean_value": float(mean_val),
            "std_value": float(std_val),
            "temporal_patterns": patterns,
            "esn_states": [esn.get_state().tolist() for esn in self.esns],
        }
    
    def reset(self) -> None:
        """Reset the temporal specialist."""
        for esn in self.esns:
            esn.reset_state()
        self.memory_buffer = []
        logger.info("TemporalESNSpecialist reset")
