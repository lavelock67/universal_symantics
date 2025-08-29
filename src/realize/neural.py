from .base import Realizer, RealizeConfig
from typing import Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)

class NeuralRealizer(Realizer):
    def __init__(self, backend):
        self.backend = backend  # Marian/M2M/NLLB

    def realize(self, src_eil, tgt_lang: str, *, binder=None, config: RealizeConfig = RealizeConfig()):
        """
        Neural realization with unified interface.
        
        Args:
            src_eil: Source EIL graph
            tgt_lang: Target language code
            binder: Optional glossary binder
            config: Realization configuration
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            # 1) expand molecules if you pivot via Minimal English (optional)
            expanded_eil = self._expand_molecules(src_eil)
            
            # 2) apply binder (preserve/gloss policy)
            if binder:
                expanded_eil = self._apply_binder(expanded_eil, binder)
            
            # 3) generate (respect config.constraints if supported)
            text = self.backend.generate(expanded_eil, tgt_lang, binder=binder, config=config)
            
            generation_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(f"Neural realization completed in {generation_time:.1f}ms")
            
            return text
            
        except Exception as e:
            logger.error(f"Neural realization failed: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def _expand_molecules(self, eil_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Expand molecules to Minimal English."""
        # This would expand complex molecules to simpler NSM prime combinations
        # For now, return the original graph
        return eil_graph
    
    def _apply_binder(self, eil_graph: Dict[str, Any], binder) -> Dict[str, Any]:
        """Apply glossary binding to preserve domain terms."""
        # Extract text from EIL graph
        text = self._extract_text_from_eil(eil_graph)
        
        # Apply binder preservation
        if hasattr(binder, 'preserve_terms'):
            for term in binder.preserve_terms:
                text = self._enforce_exact_form(text, term)
        
        # Apply binder glossing
        if hasattr(binder, 'gloss_terms'):
            for term in binder.gloss_terms:
                text = self._inject_gloss(text, term)
        
        # Update EIL graph
        updated_eil = eil_graph.copy()
        updated_eil["preserved_text"] = text
        return updated_eil
    
    def _extract_text_from_eil(self, eil_graph: Dict[str, Any]) -> str:
        """Extract text from EIL graph."""
        # Simplified extraction - would be more sophisticated in practice
        primes = eil_graph.get("primes", [])
        if isinstance(primes, list):
            return " ".join([str(p) for p in primes])
        return str(eil_graph)
    
    def _enforce_exact_form(self, text: str, term: str) -> str:
        """Enforce exact form of a term in text."""
        # Simplified implementation - would use more sophisticated matching
        return text.replace(term.lower(), term)
    
    def _inject_gloss(self, text: str, term: str) -> str:
        """Inject gloss for a term."""
        # Simplified implementation - would add Minimal English gloss
        return text.replace(term, f"{term} (gloss)")
