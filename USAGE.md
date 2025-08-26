# NSM Universal Translator - Usage Guide

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/lavelock67/universal_symantics.git
cd universal_symantics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm

# Copy configuration
cp config.example.env .env
# Edit .env with your settings
```

### 2. Start the API Server

```bash
# Start the API server
python api/enhanced_nsm_api.py

# Or with Docker (if available)
docker-compose up -d
```

The API will be available at `http://localhost:8001`

## API Endpoints

### 1. Basic Detection

**Detect NSM primes in text:**

```bash
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "La gente piensa que esto es muy bueno",
    "language": "es"
  }'
```

**Response:**
```json
{
  "primes": ["PEOPLE", "THINK", "THIS", "VERY", "GOOD"],
  "language": "es",
  "processing_time": 0.005,
  "method_results": {
    "ud_patterns": ["VERY", "THIS", "THINK", "PEOPLE"],
    "mwe_detection": [],
    "lexical_patterns": ["HasProperty", "PEOPLE", "THINK", "GOOD", "VERY", "THIS"]
  }
}
```

### 2. Round-trip Translation

**Translate with fidelity checking:**

```bash
curl -X POST "http://localhost:8001/roundtrip" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "La gente piensa que esto es muy bueno",
    "source_language": "es",
    "target_language": "fr",
    "constraint_mode": "hybrid",
    "realizer": "strict"
  }'
```

**Response:**
```json
{
  "explication_graph": {
    "primes": ["PEOPLE", "THINK", "THIS", "VERY", "GOOD"],
    "structure": "THINK(PEOPLE, GOOD(VERY(THIS)))"
  },
  "target_text": "Les gens pensent que c'est très bon",
  "legality": 0.95,
  "molecule_ratio": 0.0,
  "drift": {
    "graph_f1": 0.92,
    "precision": 0.95,
    "recall": 0.90,
    "coverage": 0.85
  },
  "router": {
    "decision": "translate",
    "risk": 0.08,
    "confidence": 0.85
  },
  "trace_id": "rt_7f3a2b1c"
}
```

### 3. Constraint Ablation

**Compare different constraint modes:**

```bash
curl -X POST "http://localhost:8001/ablation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "At most half of the students read a lot of books",
    "modes": ["off", "hybrid", "hard"]
  }'
```

**Response:**
```json
{
  "runs": [
    {
      "mode": "off",
      "legality": 0.42,
      "drift": {"graph_f1": 0.51},
      "latency_ms": 220
    },
    {
      "mode": "hybrid",
      "legality": 0.86,
      "drift": {"graph_f1": 0.81},
      "latency_ms": 315
    },
    {
      "mode": "hard",
      "legality": 0.96,
      "drift": {"graph_f1": 0.88},
      "latency_ms": 360
    }
  ]
}
```

### 4. Language Assets Debug

**Check loaded language assets:**

```bash
curl -X GET "http://localhost:8001/debug/lang_assets"
```

**Response:**
```json
{
  "en": {
    "ud_model": "en_core_web_sm",
    "mwe_rules": 26,
    "exponent_entries": 65,
    "detectors": 8
  },
  "es": {
    "ud_model": "es_core_news_sm",
    "mwe_rules": 32,
    "exponent_entries": 65,
    "detectors": 8
  },
  "fr": {
    "ud_model": "fr_core_news_sm",
    "mwe_rules": 32,
    "exponent_entries": 65,
    "detectors": 8
  }
}
```

### 5. MWE Detection

**Detect multi-word expressions:**

```bash
curl -X POST "http://localhost:8001/mwe" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Au plus la moitié des élèves lisent beaucoup",
    "language": "fr"
  }'
```

**Response:**
```json
{
  "mwes": [
    {
      "phrase": "au plus",
      "prime": "NOT/MORE",
      "type": "quantifier",
      "scope": "quant"
    },
    {
      "phrase": "la moitié",
      "prime": "HALF",
      "type": "quantifier",
      "scope": "quant"
    },
    {
      "phrase": "beaucoup",
      "prime": "MANY",
      "type": "quantifier",
      "scope": "quant"
    }
  ]
}
```

## Examples by Language

### English Examples

```bash
# Everyday sentence
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "I think you know the truth", "language": "en"}'

# Quantifier scope
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "At most half of the students read a lot", "language": "en"}'

# Negation flip
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "It is not false that this works", "language": "en"}'
```

### Spanish Examples

```bash
# Everyday sentence
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "La gente piensa que esto es muy bueno", "language": "es"}'

# Negation scope
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Es falso que el medicamento no funcione", "language": "es"}'

# Quantifier scope
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "A lo sumo la mitad de los estudiantes", "language": "es"}'
```

### French Examples

```bash
# Everyday sentence
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Les gens pensent que c'est très bon", "language": "fr"}'

# Negative polarity
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Personne ne vient", "language": "fr"}'

# Quantifier scope
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Au plus la moitié des élèves lisent beaucoup", "language": "fr"}'
```

## Testing

### Run Curated Test Suite

```bash
# Run all tests
pytest tests/test_curated_suite.py -v

# Run specific test categories
pytest tests/test_curated_suite.py::TestCuratedSuite::test_safety_critical_cases -v
pytest tests/test_curated_suite.py::TestCuratedSuite::test_everyday_cases -v
```

### Run Individual Tests

```bash
# Test detection accuracy
python test_integrated_detector.py

# Test critical fixes
python test_critical_fixes.py

# Test risk router
python test_risk_router.py
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8001

# Router Thresholds (per language)
EN_LEGALITY_THRESHOLD=0.9
ES_LEGALITY_THRESHOLD=0.85
FR_LEGALITY_THRESHOLD=0.85

# Safety-Critical Feature Weights
NEGATION_SCOPE_WEIGHT=0.4
QUANTIFIER_SCOPE_WEIGHT=0.3

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Router Decision Logic

The router makes decisions based on:

1. **Safety-critical features** (negation/quantifier scope)
2. **Language-specific thresholds**
3. **Risk factors** (legality, coverage, sense confidence)
4. **Weighted risk scoring**

**Decision thresholds:**
- `risk ≤ 0.2` → **TRANSLATE**
- `0.2 < risk ≤ 0.6` → **CLARIFY**
- `risk > 0.6` → **ABSTAIN**

## Monitoring

### Prometheus Metrics

Available metrics at `http://localhost:9090`:

- `router_translate_rate{lang="en"}`
- `router_abstain_rate{lang="es"}`
- `safety_override_count{feature="negation_scope"}`
- `detection_accuracy{lang="fr"}`

### Grafana Dashboards

Access dashboards at `http://localhost:3000`:

- Router decision distribution
- Per-language performance
- Safety override tracking
- Detection accuracy trends

## Troubleshooting

### Common Issues

1. **spaCy models not found:**
   ```bash
   python -m spacy download en_core_web_sm es_core_news_sm fr_core_news_sm
   ```

2. **Port already in use:**
   ```bash
   # Change port in .env
   API_PORT=8002
   ```

3. **Memory issues:**
   ```bash
   # Reduce workers in .env
   API_WORKERS=2
   ```

### Debug Endpoints

```bash
# Check language assets
curl http://localhost:8001/debug/lang_assets

# Health check
curl http://localhost:8001/health

# Router statistics
curl http://localhost:8001/metrics
```

## Performance

### Expected Performance

- **Detection latency**: <10ms per request
- **Router latency**: <1ms per request
- **Round-trip translation**: <100ms per request
- **Memory usage**: ~2GB with all models loaded

### Optimization Tips

1. **Use smaller spaCy models** for production
2. **Enable caching** for repeated requests
3. **Adjust worker count** based on CPU cores
4. **Monitor memory usage** with large batches

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the test suite for examples
3. Check the debug endpoints
4. Open an issue on GitHub

## License

MIT License - see LICENSE file for details.
