#!/usr/bin/env python3
"""
Diagnostic script to check language asset loading
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_ud_models():
    """Check UD models."""
    print("🔍 Checking UD Models...")
    try:
        import spacy
        model_mapping = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm", 
            "fr": "fr_core_news_sm"
        }
        for lang, model_name in model_mapping.items():
            try:
                nlp = spacy.load(model_name)
                print(f"  ✅ {lang}: loaded ({model_name})")
            except Exception as e:
                print(f"  ❌ {lang}: not loaded ({model_name}) - {e}")
    except Exception as e:
        print(f"  ❌ spacy not available: {e}")

def check_mwe_tagger():
    """Check MWE tagger."""
    print("\n🔍 Checking MWE Tagger...")
    try:
        from src.detect.mwe_tagger import MWETagger
        mwe_tagger = MWETagger()
        print(f"  ✅ MWE tagger initialized")
        
        # Check lexicons
        if hasattr(mwe_tagger, 'lexicons'):
            for lang in ["en", "es", "fr"]:
                count = len(mwe_tagger.lexicons.get(lang, {}))
                print(f"  📊 {lang}: {count} MWE rules")
        else:
            print("  ❌ No lexicons attribute")
    except Exception as e:
        print(f"  ❌ MWE tagger error: {e}")

def check_exponent_lexicon():
    """Check exponent lexicon."""
    print("\n🔍 Checking Exponent Lexicon...")
    try:
        from src.detect.exponent_lexicons import ExponentLexicon
        exponent_lexicon = ExponentLexicon()
        print(f"  ✅ Exponent lexicon initialized")
        
        # Check exponents
        if hasattr(exponent_lexicon, 'exponents'):
            for lang in ["en", "es", "fr"]:
                count = len(exponent_lexicon.exponents.get(lang, {}))
                print(f"  📊 {lang}: {count} exponent entries")
        else:
            print("  ❌ No exponents attribute")
    except Exception as e:
        print(f"  ❌ Exponent lexicon error: {e}")

def check_detectors():
    """Check detector registration."""
    print("\n🔍 Checking Detectors...")
    try:
        from src.detect.srl_ud_detectors import ALL_NSM_PRIMES
        print(f"  ✅ {len(ALL_NSM_PRIMES)} NSM primes available")
        
        # Check if multilingual detection is available
        from src.detect.srl_ud_detectors import detect_primitives_multilingual
        print(f"  ✅ Multilingual detection available")
    except Exception as e:
        print(f"  ❌ Detector error: {e}")

def test_detection():
    """Test actual detection."""
    print("\n🔍 Testing Detection...")
    
    test_cases = [
        ("en", "I think you know the truth about this"),
        ("es", "La gente piensa que esto es muy bueno"),
        ("fr", "Les gens pensent que c'est très bon")
    ]
    
    try:
        from src.detect.srl_ud_detectors import detect_primitives_multilingual
        
        for lang, text in test_cases:
            try:
                primes = detect_primitives_multilingual(text)
                print(f"  📝 {lang}: '{text}' → {len(primes)} primes: {primes}")
            except Exception as e:
                print(f"  ❌ {lang}: detection failed - {e}")
    except Exception as e:
        print(f"  ❌ Detection test failed: {e}")

def main():
    """Run all diagnostics."""
    print("🚀 NSM Language Asset Diagnostics")
    print("=" * 50)
    
    check_ud_models()
    check_mwe_tagger()
    check_exponent_lexicon()
    check_detectors()
    test_detection()
    
    print("\n" + "=" * 50)
    print("🎯 Diagnostic Complete")

if __name__ == "__main__":
    main()
