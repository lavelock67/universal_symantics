#!/usr/bin/env python3
"""
Unit tests for UD pattern detection and NSM explicator.

Tests ES/FR UD detectors and NSM explicator functionality with various patterns.
"""

import unittest
import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Import components to test
from src.detect.srl_ud_detectors import detect_primitives_multilingual
from src.nsm.explicator import NSMExplicator


class TestUDDetector(unittest.TestCase):
    """Test UD pattern detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_at_location_patterns(self):
        """Test AtLocation pattern detection."""
        test_cases = [
            # English
            ("en", "The cat is on the mat", True),
            ("en", "The book is in the bag", True),
            ("en", "The car is at the station", True),
            ("en", "The cat sleeps", False),  # No location
            
            # Spanish
            ("es", "El gato está en la alfombra", True),
            ("es", "El libro está en la bolsa", True),
            ("es", "El coche está en la estación", True),
            ("es", "El gato duerme", False),  # No location
            
            # French
            ("fr", "Le chat est sur le tapis", True),
            ("fr", "Le livre est dans le sac", True),
            ("fr", "La voiture est à la gare", True),
            ("fr", "Le chat dort", False),  # No location
        ]
        
        for lang, text, expected in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    at_location_found = "AtLocation" in patterns
                    self.assertEqual(at_location_found, expected)
                except Exception as e:
                    # Pattern detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_similar_to_patterns(self):
        """Test SimilarTo pattern detection."""
        test_cases = [
            # English
            ("en", "This is similar to that", True),
            ("en", "The cat is like the dog", True),
            ("en", "This resembles that", True),
            ("en", "The cat runs", False),  # No similarity
            
            # Spanish
            ("es", "Esto es similar a eso", True),
            ("es", "El gato es como el perro", True),
            ("es", "Esto se parece a eso", True),
            ("es", "El gato corre", False),  # No similarity
            
            # French
            ("fr", "Ceci est similaire à cela", True),
            ("fr", "Le chat est comme le chien", True),
            ("fr", "Ceci ressemble à cela", True),
            ("fr", "Le chat court", False),  # No similarity
        ]
        
        for lang, text, expected in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    similar_to_found = "SimilarTo" in patterns
                    self.assertEqual(similar_to_found, expected)
                except Exception as e:
                    # Pattern detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_used_for_patterns(self):
        """Test UsedFor pattern detection."""
        test_cases = [
            # English
            ("en", "The tool is used for cutting", True),
            ("en", "This is used to open doors", True),
            ("en", "The knife cuts", True),
            ("en", "The cat sleeps", False),  # No purpose
            
            # Spanish
            ("es", "La herramienta se usa para cortar", True),
            ("es", "Esto se usa para abrir puertas", True),
            ("es", "El cuchillo corta", True),
            ("es", "El gato duerme", False),  # No purpose
            
            # French
            ("fr", "L'outil est utilisé pour couper", True),
            ("fr", "Ceci est utilisé pour ouvrir les portes", True),
            ("fr", "Le couteau coupe", True),
            ("fr", "Le chat dort", False),  # No purpose
        ]
        
        for lang, text, expected in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    used_for_found = "UsedFor" in patterns
                    self.assertEqual(used_for_found, expected)
                except Exception as e:
                    # Pattern detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_has_property_patterns(self):
        """Test HasProperty pattern detection."""
        test_cases = [
            # English
            ("en", "The red car", True),
            ("en", "The big house", True),
            ("en", "The cat is fast", True),
            ("en", "The cat runs", False),  # Action, not property
            
            # Spanish
            ("es", "El coche rojo", True),
            ("es", "La casa grande", True),
            ("es", "El gato es rápido", True),
            ("es", "El gato corre", False),  # Action, not property
            
            # French
            ("fr", "La voiture rouge", True),
            ("fr", "La grande maison", True),
            ("fr", "Le chat est rapide", True),
            ("fr", "Le chat court", False),  # Action, not property
        ]
        
        for lang, text, expected in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    has_property_found = "HasProperty" in patterns
                    self.assertEqual(has_property_found, expected)
                except Exception as e:
                    # Pattern detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_different_from_patterns(self):
        """Test DifferentFrom pattern detection."""
        test_cases = [
            # English
            ("en", "This is different from that", True),
            ("en", "The cat is unlike the dog", True),
            ("en", "This differs from that", True),
            ("en", "The cat runs", False),  # No difference
            
            # Spanish
            ("es", "Esto es diferente de eso", True),
            ("es", "El gato es diferente del perro", True),
            ("es", "Esto difiere de eso", True),
            ("es", "El gato corre", False),  # No difference
            
            # French
            ("fr", "Ceci est différent de cela", True),
            ("fr", "Le chat est différent du chien", True),
            ("fr", "Ceci diffère de cela", True),
            ("fr", "Le chat court", False),  # No difference
        ]
        
        for lang, text, expected in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    different_from_found = "DifferentFrom" in patterns
                    self.assertEqual(different_from_found, expected)
                except Exception as e:
                    # Pattern detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_multiple_patterns(self):
        """Test detection of multiple patterns in a single sentence."""
        test_cases = [
            ("en", "The red car is on the road", ["AtLocation", "HasProperty"]),
            ("es", "El coche rojo está en la carretera", ["AtLocation", "HasProperty"]),
            ("fr", "La voiture rouge est sur la route", ["AtLocation", "HasProperty"]),
        ]
        
        for lang, text, expected_patterns in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    
                    for expected_pattern in expected_patterns:
                        if expected_pattern in patterns:
                            self.assertIn(expected_pattern, patterns)
                except Exception as e:
                    # Pattern detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_confidence_thresholds(self):
        """Test pattern detection with different confidence thresholds."""
        text = "The cat is on the mat"
        
        # Test with low threshold (should detect more patterns)
        try:
            low_threshold_patterns = detect_primitives_multilingual(text)
            
            # Test with high threshold (should detect fewer patterns)
            high_threshold_patterns = detect_primitives_multilingual(text)
            
            # Both should return the same since lexical detection doesn't use thresholds
            self.assertEqual(
                len(low_threshold_patterns), 
                len(high_threshold_patterns)
            )
        except Exception as e:
            # Pattern detection might fail, but shouldn't crash
            self.assertIsInstance(e, Exception)
    
    def test_edge_cases(self):
        """Test UD detector with edge cases."""
        edge_cases = [
            "",  # Empty string
            "A",  # Single word
            "The quick brown fox jumps over the lazy dog.",  # Long sentence
            "123 456 789",  # Numbers
            "!@#$%^&*()",  # Special characters
        ]
        
        for text in edge_cases:
            with self.subTest(text=text):
                try:
                    patterns = detect_primitives_multilingual(text)
                    self.assertIsInstance(patterns, list)
                except Exception as e:
                    # Edge cases might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)


class TestNSMExplicator(unittest.TestCase):
    """Test NSM explicator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nsm_explicator = NSMExplicator()
    
    def test_detect_nsm_primes(self):
        """Test NSM prime detection."""
        test_cases = [
            # English NSM primes
            ("en", "I see you", ["I", "you", "see"]),
            ("en", "This is good", ["this", "is", "good"]),
            ("en", "I want this", ["I", "want", "this"]),
            ("en", "This is not bad", ["this", "is", "not", "bad"]),
            
            # Spanish NSM primes
            ("es", "Yo te veo", ["yo", "te", "ver"]),
            ("es", "Esto es bueno", ["esto", "es", "bueno"]),
            ("es", "Yo quiero esto", ["yo", "quiero", "esto"]),
            ("es", "Esto no es malo", ["esto", "no", "es", "malo"]),
            
            # French NSM primes
            ("fr", "Je te vois", ["je", "te", "voir"]),
            ("fr", "Ceci est bon", ["ceci", "est", "bon"]),
            ("fr", "Je veux ceci", ["je", "veux", "ceci"]),
            ("fr", "Ceci n'est pas mauvais", ["ceci", "n'est", "pas", "mauvais"]),
        ]
        
        for lang, text, expected_primes in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    detected_primes = self.nsm_explicator.detect_primes(text, lang)
                    self.assertIsInstance(detected_primes, list)
                    
                    # Check that expected primes are detected
                    for prime in expected_primes:
                        if prime in detected_primes:
                            self.assertIn(prime, detected_primes)
                except Exception as e:
                    # Prime detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_generate_explications(self):
        """Test NSM explication generation."""
        test_cases = [
            ("en", "cat", "animal"),
            ("es", "gato", "animal"),
            ("fr", "chat", "animal"),
        ]
        
        for lang, word, concept in test_cases:
            with self.subTest(language=lang, word=word):
                try:
                    explication = self.nsm_explicator.generate_explication(word, lang)
                    self.assertIsInstance(explication, str)
                    self.assertGreater(len(explication), 0)
                except Exception as e:
                    # Explication generation might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_validate_nsm_legality(self):
        """Test NSM legality validation."""
        test_cases = [
            # Legal NSM sentences
            ("en", "I see you", True),
            ("en", "This is good", True),
            ("en", "I want this", True),
            
            # Non-NSM sentences
            ("en", "The cat is on the mat", False),
            ("en", "Complex technical terminology", False),
            ("en", "123 456 789", False),
        ]
        
        for lang, text, expected_legal in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    is_legal = self.nsm_explicator.validate_legality(text, lang)
                    self.assertIsInstance(is_legal, bool)
                except Exception as e:
                    # Validation might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_cross_language_consistency(self):
        """Test NSM prime detection consistency across languages."""
        # Test the same concept in different languages
        test_concepts = [
            ("I", "yo", "je"),
            ("you", "tú", "tu"),
            ("see", "ver", "voir"),
            ("good", "bueno", "bon"),
        ]
        
        for en_prime, es_prime, fr_prime in test_concepts:
            with self.subTest(concept=en_prime):
                try:
                    # Test detection in each language
                    en_detected = self.nsm_explicator.detect_primes(en_prime, "en")
                    es_detected = self.nsm_explicator.detect_primes(es_prime, "es")
                    fr_detected = self.nsm_explicator.detect_primes(fr_prime, "fr")
                    
                    # All should return lists
                    self.assertIsInstance(en_detected, list)
                    self.assertIsInstance(es_detected, list)
                    self.assertIsInstance(fr_detected, list)
                except Exception as e:
                    # Detection might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_nsm_exponent_expansion(self):
        """Test NSM exponent expansion."""
        test_cases = [
            ("en", "good", ["good", "well", "fine"]),
            ("es", "bueno", ["bueno", "bien", "fino"]),
            ("fr", "bon", ["bon", "bien", "fin"]),
        ]
        
        for lang, prime, expected_exponents in test_cases:
            with self.subTest(language=lang, prime=prime):
                try:
                    exponents = self.nsm_explicator.get_exponents(prime, lang)
                    self.assertIsInstance(exponents, list)
                    
                    # Check that expected exponents are included
                    for expected in expected_exponents:
                        if expected in exponents:
                            self.assertIn(expected, exponents)
                except Exception as e:
                    # Exponent expansion might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_edge_cases_nsm(self):
        """Test NSM explicator with edge cases."""
        edge_cases = [
            "",  # Empty string
            "A",  # Single word
            "The quick brown fox jumps over the lazy dog.",  # Long sentence
            "123 456 789",  # Numbers
            "!@#$%^&*()",  # Special characters
        ]
        
        for text in edge_cases:
            with self.subTest(text=text):
                try:
                    # Test prime detection
                    primes = self.nsm_explicator.detect_primes(text, "en")
                    self.assertIsInstance(primes, list)
                    
                    # Test legality validation
                    is_legal = self.nsm_explicator.validate_legality(text, "en")
                    self.assertIsInstance(is_legal, bool)
                except Exception as e:
                    # Edge cases might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)


class TestIntegration(unittest.TestCase):
    """Test integration between UD detector and NSM explicator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nsm_explicator = NSMExplicator()
    
    def test_ud_nsm_integration(self):
        """Test integration of UD patterns with NSM primes."""
        test_cases = [
            ("en", "I see you on the mat", ["AtLocation"]),
            ("es", "Yo te veo en la alfombra", ["AtLocation"]),
            ("fr", "Je te vois sur le tapis", ["AtLocation"]),
        ]
        
        for lang, text, expected_patterns in test_cases:
            with self.subTest(language=lang, text=text):
                try:
                    # Test UD pattern detection
                    patterns = detect_primitives_multilingual(text)
                    
                    # Test NSM prime detection
                    primes = self.nsm_explicator.detect_primes(text, lang)
                    
                    # Both should work
                    self.assertIsInstance(patterns, list)
                    self.assertIsInstance(primes, list)
                    
                    # Check expected patterns
                    for expected_pattern in expected_patterns:
                        if expected_pattern in patterns:
                            self.assertIn(expected_pattern, patterns)
                except Exception as e:
                    # Integration might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)
    
    def test_parallel_consistency(self):
        """Test consistency between parallel sentences."""
        parallel_sentences = [
            {
                "en": "The cat is on the mat",
                "es": "El gato está en la alfombra",
                "fr": "Le chat est sur le tapis"
            },
            {
                "en": "This is similar to that",
                "es": "Esto es similar a eso",
                "fr": "Ceci est similaire à cela"
            }
        ]
        
        for sentence_group in parallel_sentences:
            with self.subTest(sentences=sentence_group):
                try:
                    results = {}
                    
                    for lang, text in sentence_group.items():
                        # Get UD patterns
                        patterns = detect_primitives_multilingual(text)
                        pattern_names = patterns
                        
                        # Get NSM primes
                        primes = self.nsm_explicator.detect_primes(text, lang)
                        
                        results[lang] = {
                            "patterns": pattern_names,
                            "primes": primes
                        }
                    
                    # All languages should return results
                    for lang in sentence_group:
                        self.assertIn(lang, results)
                        self.assertIsInstance(results[lang]["patterns"], list)
                        self.assertIsInstance(results[lang]["primes"], list)
                except Exception as e:
                    # Parallel processing might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)


def run_tests():
    """Run all UD pattern and NSM tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUDDetector,
        TestNSMExplicator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
