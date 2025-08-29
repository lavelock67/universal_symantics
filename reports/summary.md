# Universal Translator Evaluation Report

**Timestamp**: 2025-08-29T07:50:33.746725
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

## Failures (45 total)

- **NEAR_es_01** (prime): "Attempt to overwrite 'exc_info' in LogRecord"
- **NEAR_es_01** (prime): Test failed
- **NEAR_es_01** (prime): Test failed
- **INSIDE_es_01** (prime): "Attempt to overwrite 'exc_info' in LogRecord"
- **INSIDE_es_01** (prime): Test failed
- **INSIDE_es_01** (prime): Test failed
- **ABOVE_fr_01** (prime): "Attempt to overwrite 'exc_info' in LogRecord"
- **ABOVE_fr_01** (prime): Test failed
- **ABOVE_fr_01** (prime): Test failed
- **ONE_fr_01** (prime): "Attempt to overwrite 'exc_info' in LogRecord"
