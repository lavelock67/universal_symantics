#!/usr/bin/env python3
"""
Debug word detection specifically
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_word_detection():
    """Test word detection specifically"""
    try:
        print("🔍 Testing word detection...")
        
        from src.core.application.services import NSMDetectionService
        from src.core.domain.models import Language
        import spacy
        
        # Create service
        service = NSMDetectionService()
        print("✅ Service created")
        
        # Test SpaCy tokenization
        print("📝 Testing SpaCy tokenization...")
        spacy_model = service.spacy_models.get(Language.ENGLISH.value)
        doc = spacy_model("word")
        print(f"✅ Token: '{doc[0].text}' -> lemma: '{doc[0].lemma_}' -> pos: '{doc[0].pos_}'")
        
        # Test pattern matching
        print("📝 Testing pattern matching...")
        patterns = service._get_lexical_patterns(Language.ENGLISH)
        word_pattern = patterns.get("WORDS")
        print(f"✅ WORDS pattern: {word_pattern}")
        
        if word_pattern:
            matches = service._matches_pattern(doc[0], word_pattern)
            print(f"✅ Pattern matches: {matches}")
        
        # Test full detection
        print("📝 Testing full detection...")
        result = service.detect_primes("word", Language.ENGLISH)
        print(f"✅ Detection result: {[p.text for p in result.primes]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_word_detection()
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")
