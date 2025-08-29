"""
NSM Prime Generator
Converts semantic decomposition (UD + SRL) to proper NSM primes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class NSMPrime:
    """NSM Prime representation"""
    prime: str
    arguments: List[str] = None
    confidence: float = 1.0
    source: str = "semantic_decomposition"

class NSMGenerator:
    """Generate NSM primes from semantic decomposition"""
    
    def __init__(self):
        # Semantic role to NSM prime mappings
        self.role_to_prime = {
            'AGENT': 'SOMEONE',
            'PATIENT': 'SOMETHING', 
            'THEME': 'SOMETHING',
            'LOCATION': 'PLACE',
            'GOAL': 'PLACE',
            'SOURCE': 'PLACE',
            'TIME': 'TIME',
            'MANNER': 'WAY',
            'INSTRUMENT': 'SOMETHING',
            'CAUSE': 'SOMETHING',
            'PURPOSE': 'SOMETHING'
        }
        
        # Spatial expressions to NSM primes
        self.spatial_to_prime = {
            'near': 'NEAR',
            'cerca': 'NEAR', 
            'près': 'NEAR',
            'inside': 'INSIDE',
            'dentro': 'INSIDE',
            'dans': 'INSIDE',
            'above': 'ABOVE',
            'encima': 'ABOVE',
            'au-dessus': 'ABOVE',
            'over': 'ABOVE',
            'sobre': 'ABOVE',
            'sur': 'ABOVE'
        }
        
        # Temporal expressions to NSM primes
        self.temporal_to_prime = {
            'now': 'NOW',
            'ahora': 'NOW',
            'maintenant': 'NOW',
            'before': 'BEFORE',
            'antes': 'BEFORE', 
            'avant': 'BEFORE',
            'after': 'AFTER',
            'después': 'AFTER',
            'après': 'AFTER'
        }
        
        # Negation markers
        self.negation_markers = {
            'not', 'no', 'ne', 'pas', 'nicht', 'non'
        }
    
    def generate_primes(self, semantic_decomposition: Dict[str, Any]) -> List[NSMPrime]:
        """Generate NSM primes from semantic decomposition"""
        primes = []
        
        structure = semantic_decomposition.get('enhanced_structure')
        if not structure:
            return primes
        
        # Generate primes from semantic roles
        for role in structure.semantic_roles:
            prime = self._role_to_prime(role)
            if prime:
                primes.append(prime)
        
        # Generate primes from spatial expressions
        for spatial_expr in structure.spatial_expressions:
            prime = self._spatial_to_prime(spatial_expr)
            if prime:
                primes.append(prime)
        
        # Generate primes from temporal expressions
        for temporal_expr in structure.temporal_expressions:
            prime = self._temporal_to_prime(temporal_expr)
            if prime:
                primes.append(prime)
        
        # Generate negation prime
        if structure.negation:
            primes.append(NSMPrime(prime='NOT', confidence=0.9, source='negation'))
        
        # Generate basic action prime
        if structure.predicate or any(role.role == 'AGENT' for role in structure.semantic_roles):
            primes.append(NSMPrime(prime='DO', confidence=0.8, source='action'))
        
        return primes
    
    def _role_to_prime(self, role) -> Optional[NSMPrime]:
        """Convert semantic role to NSM prime"""
        prime_name = self.role_to_prime.get(role.role)
        if prime_name:
            return NSMPrime(
                prime=prime_name,
                arguments=[role.text],
                confidence=role.confidence,
                source=f"semantic_role:{role.role}"
            )
        return None
    
    def _spatial_to_prime(self, spatial_expr: str) -> Optional[NSMPrime]:
        """Convert spatial expression to NSM prime"""
        spatial_lower = spatial_expr.lower()
        for pattern, prime_name in self.spatial_to_prime.items():
            if pattern in spatial_lower:
                return NSMPrime(
                    prime=prime_name,
                    arguments=[spatial_expr],
                    confidence=0.9,
                    source=f"spatial:{pattern}"
                )
        return None
    
    def _temporal_to_prime(self, temporal_expr: str) -> Optional[NSMPrime]:
        """Convert temporal expression to NSM prime"""
        temporal_lower = temporal_expr.lower()
        for pattern, prime_name in self.temporal_to_prime.items():
            if pattern in temporal_lower:
                return NSMPrime(
                    prime=prime_name,
                    arguments=[temporal_expr],
                    confidence=0.9,
                    source=f"temporal:{pattern}"
                )
        return None
