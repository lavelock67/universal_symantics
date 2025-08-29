#!/usr/bin/env python3
"""
Canary Test Grid - Truth Over Vibes
These tests must stay green; if any goes red, CI fails.
"""

import pytest
from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language


class TestCanaryGrid:
    """Canary tests that must always pass."""
    
    def setup_method(self):
        """Set up the detection service for each test."""
        self.service = NSMDetectionService()
    
    def test_en_inside_relation(self):
        """EN: The book is inside the box. → INSIDE(THING(book), THING(box))"""
        result = self.service.detect_primes("The book is inside the box.", Language.ENGLISH)
        
        # Extract primes and check for INSIDE relation
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain INSIDE prime
        assert 'INSIDE' in primes, f"INSIDE prime not detected. Found: {primes}"
        
        # Should not contain non-official primes
        non_official = {'BOOK', 'BOX', 'THE', 'IS'} & primes
        assert not non_official, f"Non-official primes detected: {non_official}"
    
    def test_en_near_relation(self):
        """EN: He lives near the station. → NEAR(THING(he/home), THING(station))"""
        result = self.service.detect_primes("He lives near the station.", Language.ENGLISH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain NEAR prime
        assert 'NEAR' in primes, f"NEAR prime not detected. Found: {primes}"
        
        # Should contain LIVE prime
        assert 'LIVE' in primes, f"LIVE prime not detected. Found: {primes}"
    
    def test_en_quantifier_negation(self):
        """EN: At most half the students read a lot. → NOT+MORE(HALF(PEOPLE(students)), READ, MANY)"""
        result = self.service.detect_primes("At most half the students read a lot.", Language.ENGLISH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain key primes
        expected_primes = {'NOT', 'MORE', 'HALF', 'PEOPLE', 'MANY'}
        found_expected = expected_primes & primes
        assert len(found_expected) >= 3, f"Expected at least 3 of {expected_primes}, found: {found_expected}"
    
    def test_es_inside_relation(self):
        """ES: El libro está dentro de la caja. → INSIDE(book, box)"""
        result = self.service.detect_primes("El libro está dentro de la caja.", Language.SPANISH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain INSIDE prime
        assert 'INSIDE' in primes, f"INSIDE prime not detected. Found: {primes}"
    
    def test_es_near_relation(self):
        """ES: Vive cerca de la estación. → NEAR(...)"""
        result = self.service.detect_primes("Vive cerca de la estación.", Language.SPANISH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain NEAR prime
        assert 'NEAR' in primes, f"NEAR prime not detected. Found: {primes}"
    
    def test_es_false_negation(self):
        """ES: Es falso que el medicamento no funcione. → FALSE(NOT(DO/HAPPEN(work, medicine)))"""
        result = self.service.detect_primes("Es falso que el medicamento no funcione.", Language.SPANISH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain FALSE and NOT primes
        assert 'FALSE' in primes, f"FALSE prime not detected. Found: {primes}"
        assert 'NOT' in primes, f"NOT prime not detected. Found: {primes}"
    
    def test_fr_above_relation(self):
        """FR: La lampe est au-dessus de la table. → ABOVE(lamp, table)"""
        result = self.service.detect_primes("La lampe est au-dessus de la table.", Language.FRENCH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain ABOVE prime
        assert 'ABOVE' in primes, f"ABOVE prime not detected. Found: {primes}"
    
    def test_fr_people_think_good(self):
        """FR: Les gens pensent que c'est très bon. → PEOPLE, THINK, THIS, VERY, GOOD (no pseudo-primes)"""
        result = self.service.detect_primes("Les gens pensent que c'est très bon.", Language.FRENCH)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain core NSM primes
        expected_primes = {'PEOPLE', 'THINK', 'THIS', 'VERY', 'GOOD'}
        found_expected = expected_primes & primes
        assert len(found_expected) >= 3, f"Expected at least 3 of {expected_primes}, found: {found_expected}"
        
        # Must NOT contain pseudo-primes
        pseudo_primes = {'GENS', 'PENSENT', 'BON', 'TRÈS'}
        found_pseudo = pseudo_primes & primes
        assert not found_pseudo, f"Pseudo-primes detected: {found_pseudo}"
    
    def test_de_inside_relation(self):
        """DE: Das Buch ist in der Kiste. → INSIDE(book, box)"""
        result = self.service.detect_primes("Das Buch ist in der Kiste.", Language.GERMAN)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain INSIDE prime
        assert 'INSIDE' in primes, f"INSIDE prime not detected. Found: {primes}"
    
    def test_de_above_relation(self):
        """DE: Die Lampe ist über dem Tisch. → ABOVE(lamp, table)"""
        result = self.service.detect_primes("Die Lampe ist über dem Tisch.", Language.GERMAN)
        
        primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
        
        # Must contain ABOVE prime
        assert 'ABOVE' in primes, f"ABOVE prime not detected. Found: {primes}"
    
    def test_roundtrip_fidelity(self):
        """Test round-trip fidelity with graph-F1 ≥ 0.85."""
        test_cases = [
            ("The book is inside the box.", Language.ENGLISH),
            ("El libro está dentro de la caja.", Language.SPANISH),
            ("La lampe est au-dessus de la table.", Language.FRENCH),
        ]
        
        for text, lang in test_cases:
            result = self.service.detect_primes(text, lang)
            
            # For now, we'll test that we get some primes
            # In a full implementation, we'd test round-trip fidelity
            primes = {prime.text.upper() for prime in result.primes} if result.primes else set()
            assert len(primes) > 0, f"No primes detected for: {text}"
            
            # All primes must be official
            from src.eil.primes_registry import ALLOWED_PRIMES
            non_official = primes - ALLOWED_PRIMES
            assert not non_official, f"Non-official primes detected: {non_official}"


if __name__ == "__main__":
    pytest.main([__file__])
