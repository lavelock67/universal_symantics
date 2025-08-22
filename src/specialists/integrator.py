"""Central integration hub for cross-modal information processing.

This module implements the central workspace and attention router mentioned in the plan,
providing cross-modal integration with Φ (phi) computation for measuring integration.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from ..table.algebra import PrimitiveAlgebra
from ..table.schema import PeriodicTable, Primitive
from .temporal_esn import TemporalESNSpecialist

logger = logging.getLogger(__name__)


class CentralHub:
    """Central integration hub for cross-modal information processing.
    
    This class implements the integration hub from the plan: central workspace +
    attention router, broadcasting mechanism, conflict detection, and Φ computation.
    """
    
    def __init__(self, periodic_table: PeriodicTable, 
                 temporal_specialist: Optional[TemporalESNSpecialist] = None):
        """Initialize the central integration hub.
        
        Args:
            periodic_table: The periodic table of primitives
            temporal_specialist: Optional temporal ESN specialist
        """
        self.table = periodic_table
        self.algebra = PrimitiveAlgebra(periodic_table)
        self.temporal_specialist = temporal_specialist
        
        # Integration workspace
        self.workspace: Dict[str, Any] = {}
        self.attention_weights: Dict[str, float] = {}
        self.conflict_history: List[Dict[str, Any]] = []
        
        # Broadcasting channels
        self.broadcast_channels: Dict[str, List[Any]] = {}
        
        # Integration metrics
        self.integration_history: List[float] = []
        self.phi_history: List[float] = []
        
        # Cross-modal connections
        self.cross_modal_connections: Dict[Tuple[str, str], float] = {}
        
        logger.info("Initialized CentralHub for cross-modal integration")
    
    def integrate(self, signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Integrate signals from multiple modalities.
        
        Args:
            signals: Dictionary mapping modality names to signal arrays
            
        Returns:
            Dictionary with integration results and Φ computation
        """
        logger.info(f"Integrating signals from {len(signals)} modalities")
        
        # Update workspace with new signals
        self._update_workspace(signals)
        
        # Compute attention weights
        attention_weights = self._compute_attention_weights(signals)
        self.attention_weights.update(attention_weights)
        
        # Detect conflicts
        conflicts = self._detect_conflicts(signals)
        if conflicts:
            self.conflict_history.extend(conflicts)
        
        # Compute cross-modal connections
        connections = self._compute_cross_modal_connections(signals)
        self.cross_modal_connections.update(connections)
        
        # Compute integration metrics
        integration_score = self._compute_integration_score(signals)
        self.integration_history.append(integration_score)
        
        # Compute Φ (phi) - integration measure
        phi_score = self._compute_phi(signals)
        self.phi_history.append(phi_score)
        
        # Broadcast integrated information
        broadcast = self._broadcast_integrated_info(signals)
        
        # Update temporal memory if available
        if self.temporal_specialist:
            temporal_result = self._update_temporal_memory(signals)
        else:
            temporal_result = {}
        
        return {
            "integration_score": integration_score,
            "phi_score": phi_score,
            "attention_weights": attention_weights,
            "conflicts": conflicts,
            "cross_modal_connections": connections,
            "broadcast": broadcast,
            "temporal_result": temporal_result,
            "workspace_state": self._get_workspace_summary(),
        }
    
    def _update_workspace(self, signals: Dict[str, np.ndarray]) -> None:
        """Update the central workspace with new signals.
        
        Args:
            signals: Dictionary of modality signals
        """
        for modality, signal in signals.items():
            # Store signal in workspace
            self.workspace[modality] = signal.copy()
            
            # Factor signal into primitives
            primitives = self.algebra.factor(signal)
            self.workspace[f"{modality}_primitives"] = primitives
            
            # Store primitive activations
            activations = self._compute_primitive_activations(signal, primitives)
            self.workspace[f"{modality}_activations"] = activations
    
    def _compute_attention_weights(self, signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute attention weights for different modalities.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Dictionary mapping modality names to attention weights
        """
        attention_weights = {}
        
        # Compute signal strength (magnitude)
        signal_strengths = {}
        for modality, signal in signals.items():
            if signal.size > 0:
                strength = np.linalg.norm(signal)
                signal_strengths[modality] = strength
            else:
                signal_strengths[modality] = 0.0
        
        # Normalize to get attention weights
        total_strength = sum(signal_strengths.values())
        if total_strength > 0:
            for modality, strength in signal_strengths.items():
                attention_weights[modality] = strength / total_strength
        else:
            # Equal attention if no signal strength
            n_modalities = len(signals)
            for modality in signals.keys():
                attention_weights[modality] = 1.0 / n_modalities
        
        return attention_weights
    
    def _detect_conflicts(self, signals: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect conflicts between different modalities.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        if len(signals) < 2:
            return conflicts
        
        # Compare signals pairwise
        modalities = list(signals.keys())
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                signal1 = signals[mod1]
                signal2 = signals[mod2]
                
                # Compute similarity
                similarity = self._compute_signal_similarity(signal1, signal2)
                
                # Detect conflict if similarity is very low
                if similarity < 0.1:  # Threshold for conflict detection
                    conflict = {
                        "modalities": [mod1, mod2],
                        "similarity": similarity,
                        "severity": 1.0 - similarity,
                        "description": f"Low similarity between {mod1} and {mod2}"
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _compute_signal_similarity(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Compute similarity between two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure signals have same shape for comparison
        if signal1.shape != signal2.shape:
            # Reshape to 1D for comparison
            signal1_flat = signal1.flatten()
            signal2_flat = signal2.flatten()
            
            # Pad or truncate to same length
            min_len = min(len(signal1_flat), len(signal2_flat))
            signal1_flat = signal1_flat[:min_len]
            signal2_flat = signal2_flat[:min_len]
        else:
            signal1_flat = signal1.flatten()
            signal2_flat = signal2.flatten()
        
        if len(signal1_flat) == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            signal1_flat.reshape(1, -1), 
            signal2_flat.reshape(1, -1)
        )[0, 0]
        
        # Ensure result is in [0, 1]
        return max(0.0, min(1.0, similarity))
    
    def _compute_cross_modal_connections(self, signals: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
        """Compute connections between different modalities.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Dictionary mapping modality pairs to connection strengths
        """
        connections = {}
        
        if len(signals) < 2:
            return connections
        
        modalities = list(signals.keys())
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                signal1 = signals[mod1]
                signal2 = signals[mod2]
                
                # Compute connection strength
                connection_strength = self._compute_signal_similarity(signal1, signal2)
                connections[(mod1, mod2)] = connection_strength
        
        return connections
    
    def _compute_integration_score(self, signals: Dict[str, np.ndarray]) -> float:
        """Compute overall integration score.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Integration score (0-1)
        """
        if len(signals) < 2:
            return 1.0  # Perfect integration for single modality
        
        # Compute average cross-modal similarity
        similarities = []
        modalities = list(signals.keys())
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                signal1 = signals[mod1]
                signal2 = signals[mod2]
                similarity = self._compute_signal_similarity(signal1, signal2)
                similarities.append(similarity)
        
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def _compute_phi(self, signals: Dict[str, np.ndarray]) -> float:
        """Compute Φ (phi) - integration measure as proxy for consciousness.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Φ score (0-1, higher indicates more integration)
        """
        if len(signals) < 2:
            return 0.0  # No integration for single modality
        
        # Compute integration score
        integration_score = self._compute_integration_score(signals)
        
        # Compute information content
        information_content = self._compute_information_content(signals)
        
        # Compute coherence
        coherence = self._compute_coherence(signals)
        
        # Φ is product of integration, information, and coherence
        phi = integration_score * information_content * coherence
        
        # Ensure phi is non-negative
        phi = max(0.0, phi)
        
        return float(phi)
    
    def _compute_information_content(self, signals: Dict[str, np.ndarray]) -> float:
        """Compute information content of the signals.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Information content score (0-1)
        """
        if not signals:
            return 0.0
        
        # Compute entropy-like measure
        total_entropy = 0.0
        total_elements = 0
        
        for signal in signals.values():
            if signal.size > 0:
                # Compute histogram
                hist, _ = np.histogram(signal.flatten(), bins=20, density=True)
                hist = hist[hist > 0]  # Remove zero bins
                
                if len(hist) > 0:
                    # Compute entropy
                    entropy = -np.sum(hist * np.log(hist + 1e-10))
                    total_entropy += entropy
                    total_elements += 1
        
        if total_elements > 0:
            avg_entropy = total_entropy / total_elements
            # Normalize to [0, 1] (assuming max entropy around 3 for 20 bins)
            return min(1.0, avg_entropy / 3.0)
        else:
            return 0.0
    
    def _compute_coherence(self, signals: Dict[str, np.ndarray]) -> float:
        """Compute coherence of the signals.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Coherence score (0-1)
        """
        if len(signals) < 2:
            return 1.0
        
        # Compute temporal coherence if temporal specialist is available
        if self.temporal_specialist:
            # Convert signals to temporal sequence
            sequence = []
            for modality, signal in signals.items():
                sequence.append({
                    "modality": modality,
                    "signal": signal,
                    "timestamp": len(sequence)
                })
            
            coherence = self.temporal_specialist.compute_temporal_coherence(sequence)
            return coherence
        else:
            # Fallback: use cross-modal similarity as coherence proxy
            return self._compute_integration_score(signals)
    
    def _compute_primitive_activations(self, signal: np.ndarray, 
                                     primitives: List[Primitive]) -> Dict[str, float]:
        """Compute activation levels of primitives for a signal.
        
        Args:
            signal: Input signal
            primitives: List of primitives
            
        Returns:
            Dictionary mapping primitive names to activation levels
        """
        activations = {}
        
        # Without a learned mapping from signals to primitives, use conservative zeros.
        # This avoids artificial inflation of Φ and integration metrics.
        for primitive in primitives:
            activations[primitive.name] = 0.0
        
        return activations
    
    def _broadcast_integrated_info(self, signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Broadcast integrated information to all channels.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Dictionary with broadcast information
        """
        # Create integrated representation
        integrated_signal = self._create_integrated_signal(signals)
        
        # Broadcast to different channels
        broadcasts = {}
        
        # High-priority channel (for important information)
        broadcasts["high_priority"] = {
            "signal": integrated_signal,
            "phi_score": self.phi_history[-1] if self.phi_history else 0.0,
            "attention_weights": self.attention_weights.copy(),
        }
        
        # General channel (for all information)
        broadcasts["general"] = {
            "signals": signals.copy(),
            "integration_score": self.integration_history[-1] if self.integration_history else 0.0,
            "conflicts": self.conflict_history[-5:] if self.conflict_history else [],  # Last 5 conflicts
        }
        
        # Feedback channel (for system feedback)
        broadcasts["feedback"] = {
            "workspace_state": self._get_workspace_summary(),
            "temporal_memory": self._get_temporal_memory_summary(),
        }
        
        return broadcasts
    
    def _create_integrated_signal(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Create an integrated signal from multiple modalities.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Integrated signal
        """
        if not signals:
            return np.array([])
        
        # Simple integration: concatenate all signals
        integrated_parts = []
        
        for modality, signal in signals.items():
            # Normalize signal
            if signal.size > 0:
                normalized = signal / (np.linalg.norm(signal) + 1e-10)
                integrated_parts.append(normalized.flatten())
        
        if integrated_parts:
            # Concatenate all parts
            integrated = np.concatenate(integrated_parts)
            
            # Apply attention weights
            if self.attention_weights:
                # Create attention mask
                attention_mask = np.ones(len(integrated))
                start_idx = 0
                for modality, weight in self.attention_weights.items():
                    if modality in signals:
                        signal_length = signals[modality].size
                        end_idx = start_idx + signal_length
                        attention_mask[start_idx:end_idx] *= weight
                        start_idx = end_idx
                
                integrated *= attention_mask
            
            return integrated
        else:
            return np.array([])
    
    def _update_temporal_memory(self, signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Update temporal memory with new signals.
        
        Args:
            signals: Dictionary of modality signals
            
        Returns:
            Dictionary with temporal memory update results
        """
        if not self.temporal_specialist:
            return {}
        
        # Convert signals to temporal sequence
        sequence = []
        for modality, signal in signals.items():
            sequence.append({
                "modality": modality,
                "signal": signal,
                "timestamp": len(sequence)
            })
        
        # Process through temporal specialist
        result = self.temporal_specialist.process_sequence(sequence)
        
        return result
    
    def _get_workspace_summary(self) -> Dict[str, Any]:
        """Get a summary of the current workspace state.
        
        Returns:
            Dictionary with workspace summary
        """
        summary = {
            "modalities": list(self.workspace.keys()),
            "attention_weights": self.attention_weights.copy(),
            "integration_history_length": len(self.integration_history),
            "phi_history_length": len(self.phi_history),
            "conflict_count": len(self.conflict_history),
        }
        
        # Add recent metrics
        if self.integration_history:
            summary["recent_integration"] = self.integration_history[-5:]  # Last 5
        if self.phi_history:
            summary["recent_phi"] = self.phi_history[-5:]  # Last 5
        
        return summary
    
    def _get_temporal_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of temporal memory.
        
        Returns:
            Dictionary with temporal memory summary
        """
        if self.temporal_specialist:
            return self.temporal_specialist.get_memory_summary()
        else:
            return {"temporal_specialist": "not_available"}
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics.
        
        Returns:
            Dictionary with integration metrics
        """
        return {
            "current_phi": self.phi_history[-1] if self.phi_history else 0.0,
            "current_integration": self.integration_history[-1] if self.integration_history else 0.0,
            "phi_history": self.phi_history.copy(),
            "integration_history": self.integration_history.copy(),
            "attention_weights": self.attention_weights.copy(),
            "cross_modal_connections": self.cross_modal_connections.copy(),
            "conflict_count": len(self.conflict_history),
            "workspace_modalities": list(self.workspace.keys()),
        }
    
    def reset(self) -> None:
        """Reset the central hub."""
        self.workspace.clear()
        self.attention_weights.clear()
        self.conflict_history.clear()
        self.broadcast_channels.clear()
        self.integration_history.clear()
        self.phi_history.clear()
        self.cross_modal_connections.clear()
        
        if self.temporal_specialist:
            self.temporal_specialist.reset()
        
        logger.info("CentralHub reset")
