# Universal Translator Evaluation Report

**Timestamp**: 2025-08-28T21:34:49.382742
**Languages**: en, es, fr
**Modes**: standard, neural, hybrid

## Metrics Summary

### Prime Detection
- **en**: F1 = 0.000
- **es**: F1 = 0.000
- **fr**: F1 = 0.000

### Scope Accuracy
- **en**: 0.000
- **es**: 0.000
- **fr**: 0.000

### Performance
- **standard**: P95 latency = 0ms
- **neural**: P95 latency = 0ms
- **hybrid**: P95 latency = 0ms

### Violations
- **Adapter invariant violations**: 0
- **Glossary violations**: 0
- **Error rate**: 0.000

## Acceptance Gates

- **prime_coverage**: ❌ FAIL
- **scope_accuracy**: ✅ PASS
- **roundtrip_fidelity**: ✅ PASS
- **invariants_glossary**: ❌ FAIL
- **performance**: ❌ FAIL

## Failures (27 total)

- **NEAR_es_01** (prime): Test failed
- **NEAR_es_01** (prime): Test failed
- **NEAR_es_01** (prime): Test failed
- **NEGATION_es_01** (scope): Test failed
- **NEGATION_es_01** (scope): Test failed
- **NEGATION_es_01** (scope): Test failed
- **QUANTIFIER_fr_01** (scope): Test failed
- **QUANTIFIER_fr_01** (scope): Test failed
- **QUANTIFIER_fr_01** (scope): Test failed
- **IDIOM_en_01** (idiom): Test failed
