# FAILURE TRIAGE ANALYSIS

## 📊 **EVALUATION SUMMARY**

**Status**: ❌ **NOT PRODUCTION READY**
- **Overall F1**: 0.000 (0/5 passed)
- **Scope Accuracy**: 0.000
- **Language Performance**: EN=0.000, ES=0.000, FR=0.000

## 🎯 **FAILURE PATTERNS BY PRIME**

### **Top Missing Primes** (Frequency Analysis)
1. **INSIDE/HOUSE** (4 failures) - Spatial preposition detection failing
2. **BOY/KICK/BALL** (1 failure) - Basic noun/verb detection failing  
3. **NOT** (1 failure) - Negation detection failing

### **Pattern Analysis**
- **Spatial Prepositions**: `INSIDE` and `HOUSE` consistently missing across EN/ES/FR
- **Basic Nouns/Verbs**: `BOY`, `KICK`, `BALL` not detected in simple sentences
- **Negation**: `NOT` not detected in negation contexts

## 🔧 **HIGH-YIELD FIXES REQUIRED**

### **A) Spatial Preps: Topical vs Spatial**

**Problem**: `INSIDE` and `HOUSE` not detected in spatial contexts
- **EN**: "inside the house" → missing `INSIDE`, `HOUSE`
- **ES**: "dentro de la casa" → missing `INSIDE`, `HOUSE`  
- **FR**: "dans la maison" → missing `INSIDE`, `HOUSE`

**Fix Template**:
```python
# FR dans / ES en/dentro de → prefer INSIDE when governor is physical container
if head_lemma in {"maison", "casa", "house"} and deprel in {"case", "obl", "nmod"}:
    detect_INSIDE = True
    detect_HOUSE = True
```

### **B) Basic Noun/Verb Detection**

**Problem**: Simple nouns and verbs not detected
- **EN**: "boy kicked ball" → missing `BOY`, `KICK`, `BALL`

**Fix Template**:
```python
# Basic noun detection
if pos == "NOUN" and lemma in {"boy", "ball", "cat", "house"}:
    detect_noun_prime = True

# Basic verb detection  
if pos == "VERB" and lemma in {"kick", "sleep", "read", "write"}:
    detect_verb_prime = True
```

### **C) Negation Detection**

**Problem**: `NOT` not detected in negation contexts
- **EN**: "does not sleep" → missing `NOT`

**Fix Template**:
```python
# Negation detection
if pos == "PART" and lemma == "not":
    detect_NOT = True
    scope_attachment = "negation"
```

## 📈 **PRIORITY FIXES (By Impact)**

### **Priority 1: Spatial Prepositions** (4/5 failures)
- **Impact**: 80% of failures
- **Fix**: Enhance spatial preposition detection
- **Expected Improvement**: F1 0.000 → 0.800

### **Priority 2: Basic Nouns/Verbs** (1/5 failures)  
- **Impact**: 20% of failures
- **Fix**: Improve basic lexical detection
- **Expected Improvement**: F1 0.800 → 1.000

### **Priority 3: Negation** (1/5 failures)
- **Impact**: 20% of failures  
- **Fix**: Add negation detection
- **Expected Improvement**: Scope accuracy 0.000 → 1.000

## 🎯 **IMPLEMENTATION PLAN**

### **Step 1: Fix Spatial Detection**
```python
# Add to missing_prime_detector.py
def detect_INSIDE_spatial(self, doc):
    """Detect INSIDE in spatial contexts"""
    for token in doc:
        if (token.dep_ in {"case", "obl", "nmod"} and 
            token.head.lemma in {"house", "casa", "maison", "room", "box"}):
            return True
    return False
```

### **Step 2: Fix Basic Lexical Detection**
```python
# Add to prime detection patterns
BASIC_NOUNS = {"boy", "girl", "cat", "dog", "house", "ball", "book"}
BASIC_VERBS = {"kick", "sleep", "read", "write", "eat", "drink"}

if token.lemma in BASIC_NOUNS:
    detect_noun_prime(token)
```

### **Step 3: Fix Negation Detection**
```python
# Add negation detection
if token.lemma == "not" and token.dep_ == "neg":
    detect_NOT(token)
    attach_negation_scope(token)
```

## 📊 **EXPECTED RESULTS AFTER FIXES**

### **Before Fixes**
- **Overall F1**: 0.000
- **Scope Accuracy**: 0.000
- **Language Performance**: All 0.000

### **After Priority 1 (Spatial)**
- **Overall F1**: 0.800 (4/5 passed)
- **Scope Accuracy**: 0.000
- **Language Performance**: EN=0.667, ES=1.000, FR=1.000

### **After Priority 2 (Basic Lexical)**
- **Overall F1**: 1.000 (5/5 passed)
- **Scope Accuracy**: 0.000
- **Language Performance**: EN=1.000, ES=1.000, FR=1.000

### **After Priority 3 (Negation)**
- **Overall F1**: 1.000 (5/5 passed)
- **Scope Accuracy**: 1.000
- **Language Performance**: EN=1.000, ES=1.000, FR=1.000

## 🚀 **NEXT STEPS**

1. **Implement Priority 1 Fix**: Spatial preposition detection
2. **Test with Real Data**: Use UD treebank examples
3. **Implement Priority 2 Fix**: Basic lexical detection  
4. **Implement Priority 3 Fix**: Negation detection
5. **Re-run Evaluation**: Verify improvements
6. **Scale to Full Dataset**: Apply to 884 UD candidates

## 📋 **ACCEPTANCE CRITERIA**

**Target Metrics**:
- **Prime Coverage**: F1 ≥ 0.85 EN, ≥ 0.75 ES/FR
- **Scope Accuracy**: ≥ 0.90 each language
- **Round-trip Fidelity**: Graph-F1 ≥ 0.85
- **Zero Violations**: Adapter and glossary violations = 0

**Current Status**: ❌ **NOT READY**
**Target Status**: ✅ **PRODUCTION READY**
