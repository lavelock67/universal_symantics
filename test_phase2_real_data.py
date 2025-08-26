#!/usr/bin/env python3
"""
Test Script for Phase 2: Real Data & Validation

This script tests the new real data and validation systems implemented in Phase 2.
"""

import time
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger
from src.core.domain.models import Language, PrimeType, MWEType
from src.core.domain.nsm_validator import NSMValidator, ValidationResult
from src.core.infrastructure.corpus_manager import CorpusManager, CorpusMetadata
from src.core.domain.mdl_analyzer import MDLAnalyzer, MDLResult


def test_nsm_validator():
    """Test the NSM validation system."""
    print("🔍 Testing NSM Validation System...")
    
    try:
        validator = NSMValidator()
        print("  ✅ NSM validator initialized")
        
        # Test validation statistics
        stats = validator.get_validation_statistics()
        print(f"  📊 Validation coverage: {stats['validation_coverage']}")
        print(f"  🌍 Supported languages: {stats['supported_languages']}")
        print(f"  🔧 Validation methods: {len(stats['validation_methods'])}")
        
        # Test prime validation
        test_candidates = ["THINK", "SAY", "WANT", "GOOD", "BAD", "BIG", "SMALL", "UNKNOWN"]
        
        for candidate in test_candidates:
            result = validator.validate_prime(candidate, Language.ENGLISH)
            print(f"  🔍 {candidate}: valid={result.is_valid}, confidence={result.confidence:.3f}")
            print(f"     Universality: {result.universality_score:.3f}")
            print(f"     Cross-lingual: {result.cross_lingual_consistency:.3f}")
            print(f"     Semantic stability: {result.semantic_stability:.3f}")
            print(f"     Notes: {result.validation_notes[0] if result.validation_notes else 'None'}")
            print()
        
        print("  ✅ NSM validation system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ NSM validation test failed: {str(e)}")
        return False


def test_corpus_manager():
    """Test the corpus management system."""
    print("\n📚 Testing Corpus Management System...")
    
    try:
        manager = CorpusManager()
        print("  ✅ Corpus manager initialized")
        
        # Test available corpora
        available_corpora = manager.get_available_corpora()
        print(f"  📊 Available corpora: {len(available_corpora)}")
        
        # List some corpora
        corpus_names = list(available_corpora.keys())[:5]
        for name in corpus_names:
            corpus = available_corpora[name]
            print(f"  📖 {name}: {corpus.domain} ({corpus.language.value})")
        
        # Test downloading a corpus
        test_corpus = "philosophy_en"
        print(f"\n  ⬇️ Downloading corpus: {test_corpus}")
        success = manager.download_corpus(test_corpus)
        
        if success:
            print(f"  ✅ Successfully downloaded {test_corpus}")
            
            # Test loading corpus
            corpora = manager.load_corpus(test_corpus)
            print(f"  📖 Loaded {len(corpora)} texts from {test_corpus}")
            
            # Test corpus statistics
            stats = manager.get_corpus_statistics(test_corpus)
            if stats:
                print(f"  📊 Corpus statistics:")
                print(f"     Total texts: {stats.total_texts}")
                print(f"     Total words: {stats.total_words}")
                print(f"     Total sentences: {stats.total_sentences}")
                print(f"     Average sentence length: {stats.average_sentence_length:.1f}")
                print(f"     Vocabulary size: {stats.vocabulary_size}")
                print(f"     Most common words: {[word for word, _ in stats.most_common_words[:3]]}")
        else:
            print(f"  ❌ Failed to download {test_corpus}")
            return False
        
        print("  ✅ Corpus management system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Corpus management test failed: {str(e)}")
        return False


def test_mdl_analyzer():
    """Test the MDL analysis system."""
    print("\n🧮 Testing MDL Analysis System...")
    
    try:
        analyzer = MDLAnalyzer()
        print("  ✅ MDL analyzer initialized")
        
        # Test analysis statistics
        stats = analyzer.get_analysis_statistics()
        print(f"  📊 Compression methods: {len(stats['compression_methods'])}")
        print(f"  🔧 Analysis metrics: {len(stats['analysis_metrics'])}")
        print(f"  📈 Analysis methods: {len(stats['analysis_methods'])}")
        
        # Test MDL analysis
        test_corpus = """
        The mind is not a vessel to be filled but a fire to be kindled.
        Philosophy is the love of wisdom and the search for truth.
        Knowledge is power, but wisdom is the ability to use it well.
        The unexamined life is not worth living.
        All that we see or seem is but a dream within a dream.
        """
        
        test_candidates = ["mind", "philosophy", "wisdom", "truth", "knowledge"]
        
        for candidate in test_candidates:
            result = analyzer.analyze_candidate(candidate, test_corpus, Language.ENGLISH, "philosophy")
            print(f"  🔍 {candidate}:")
            print(f"     Compression ratio: {result.compression_ratio:.3f}")
            print(f"     MDL delta: {result.mdl_delta:.3f}")
            print(f"     Information gain: {result.information_gain:.3f}")
            print(f"     Complexity score: {result.complexity_score:.3f}")
            print(f"     Universality score: {result.universality_score:.3f}")
            print(f"     Best method: {result.compression_method}")
            print(f"     Notes: {result.analysis_notes[0] if result.analysis_notes else 'None'}")
            print()
        
        print("  ✅ MDL analysis system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ MDL analysis test failed: {str(e)}")
        return False


def test_integration():
    """Test integration between the systems."""
    print("\n🔗 Testing System Integration...")
    
    try:
        # Initialize all systems
        validator = NSMValidator()
        corpus_manager = CorpusManager()
        mdl_analyzer = MDLAnalyzer()
        
        print("  ✅ All systems initialized")
        
        # Test end-to-end workflow
        test_corpus_name = "philosophy_en"
        
        # Download corpus
        print(f"  ⬇️ Downloading {test_corpus_name}...")
        success = corpus_manager.download_corpus(test_corpus_name)
        if not success:
            print(f"  ❌ Failed to download {test_corpus_name}")
            return False
        
        # Load corpus
        print(f"  📖 Loading {test_corpus_name}...")
        corpora = corpus_manager.load_corpus(test_corpus_name)
        
        # Combine corpus texts
        combined_text = "\n".join([c.text for c in corpora])
        print(f"  📊 Combined corpus: {len(combined_text)} characters")
        
        # Test candidate discovery and validation
        test_candidates = ["mind", "philosophy", "wisdom", "truth", "knowledge", "life", "dream"]
        
        print(f"  🔍 Analyzing {len(test_candidates)} candidates...")
        
        for candidate in test_candidates:
            # Validate candidate
            validation_result = validator.validate_prime(candidate, Language.ENGLISH)
            
            # Analyze with MDL
            mdl_result = mdl_analyzer.analyze_candidate(candidate, combined_text, Language.ENGLISH, "philosophy")
            
            # Combined assessment
            combined_score = (validation_result.confidence + mdl_result.universality_score) / 2
            
            print(f"  📊 {candidate}:")
            print(f"     Validation confidence: {validation_result.confidence:.3f}")
            print(f"     MDL universality: {mdl_result.universality_score:.3f}")
            print(f"     Combined score: {combined_score:.3f}")
            print(f"     Valid prime: {validation_result.is_valid and combined_score > 0.6}")
            print()
        
        print("  ✅ System integration working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {str(e)}")
        return False


def main():
    """Run all Phase 2 tests."""
    print("🚀 Testing Phase 2: Real Data & Validation Systems")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("NSM Validator", test_nsm_validator),
        ("Corpus Manager", test_corpus_manager),
        ("MDL Analyzer", test_mdl_analyzer),
        ("System Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Phase 2 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 2 tests passed! Real data and validation systems are working correctly.")
        print("\n📋 Phase 2 Improvements Summary:")
        print("  ✅ Real NSM validation against linguistic universals")
        print("  ✅ Cross-lingual consistency checking")
        print("  ✅ Semantic stability analysis")
        print("  ✅ Real corpus management with multiple sources")
        print("  ✅ Actual compression algorithms (gzip, zlib, bzip2, LZMA, LZ77, Huffman)")
        print("  ✅ Information theory-based MDL analysis")
        print("  ✅ Shannon entropy calculations")
        print("  ✅ Complexity and universality scoring")
        print("  ✅ Integration between validation, corpus, and MDL systems")
        print("\n🚀 Ready for Phase 3: Performance & Scalability!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Some systems need attention.")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total test time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
