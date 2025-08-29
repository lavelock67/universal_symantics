# Universal Translator - Production Deployment

A production-ready universal translation system based on Natural Semantic Metalanguage (NSM) primes with comprehensive monitoring, observability, and quality guarantees.

## 🚀 Features

### Core Translation Capabilities
- **Multi-language Support**: 10+ languages (EN, ES, FR, DE, IT, PT, RU, ZH, JA, KO)
- **NSM Prime Detection**: 65+4 canonical primes with cross-lingual consistency
- **Semantic Decomposition**: UD + SRL enhanced semantic analysis
- **Neural Generation**: T5/BART/mT5 integration with post-check guarantees
- **Cultural Adaptation**: Context-aware translation with invariant protection

### Production Features
- **Multiple Pipeline Modes**: Standard, Neural, Hybrid, Research
- **Quality Levels**: Basic, Standard, High, Research
- **Glossary Binding**: Domain-specific term preservation
- **Batch Processing**: Parallel translation with error handling
- **Comprehensive Monitoring**: Prometheus metrics, health checks, logging

### Observability & Monitoring
- **Prometheus Metrics**: Request counts, durations, error rates, Graph-F1 scores
- **Grafana Dashboards**: Real-time visualization of system performance
- **ELK Stack**: Log aggregation and analysis (Elasticsearch, Logstash, Kibana)
- **Health Checks**: Component-level health monitoring
- **Performance Tracking**: Per-stage timing and quality metrics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Load Balancer │    │   API Gateway   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Production Pipeline                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Standard  │ │   Neural    │ │   Hybrid    │ │  Research   │ │
│  │   Pipeline  │ │  Pipeline   │ │  Pipeline   │ │  Pipeline   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Components                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Prime     │ │  Semantic   │ │  Neural     │ │  Cultural   │ │
│  │ Detection   │ │Decomposition│ │ Generation  │ │ Adaptation  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Redis     │ │ PostgreSQL  │ │ Prometheus  │ │  Grafana    │ │
│  │   Cache     │ │   Storage   │ │  Metrics    │ │  Dashboards │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB+ RAM
- 20GB+ disk space
- Python 3.11+ (for development)

### Production Deployment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd universal-translator
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your production settings
   ```

3. **Build and deploy**
   ```bash
   docker-compose -f deployment/docker-compose.yml up -d
   ```

4. **Verify deployment**
   ```bash
   # Check API health
   curl http://localhost:8000/health
   
   # Check Prometheus
   curl http://localhost:9090/-/healthy
   
   # Check Grafana
   curl http://localhost:3000/api/health
   ```

### Development Setup

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run development server**
   ```bash
   python -m uvicorn api.production_api:app --reload --host 0.0.0.0 --port 8000
   ```

## 🔧 API Usage

### Basic Translation

```python
import requests

# Simple translation
response = requests.post("http://localhost:8000/translate", json={
    "source_text": "The boy kicked the ball.",
    "source_language": "en",
    "target_language": "es",
    "mode": "hybrid",
    "quality_level": "standard"
})

result = response.json()
print(f"Translated: {result['translated_text']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Detected primes: {result['detected_primes']}")
```

### Advanced Translation with Glossary

```python
# Translation with glossary preservation
response = requests.post("http://localhost:8000/translate", json={
    "source_text": "The API connects to the database server.",
    "source_language": "en",
    "target_language": "fr",
    "mode": "neural",
    "quality_level": "high",
    "glossary_terms": {
        "API": "API",
        "database": "base de données",
        "server": "serveur"
    }
})
```

### Batch Translation

```python
# Batch translation
response = requests.post("http://localhost:8000/translate/batch", json={
    "translations": [
        {
            "source_text": "Hello world",
            "source_language": "en",
            "target_language": "es"
        },
        {
            "source_text": "Good morning",
            "source_language": "en",
            "target_language": "fr"
        }
    ],
    "parallel_processing": True
})
```

## 📊 Monitoring & Observability

### Prometheus Metrics

Key metrics available at `/metrics`:

- `translation_requests_total`: Total translation requests by language and mode
- `translation_duration_seconds`: Translation duration distribution
- `translation_errors_total`: Error counts by type
- `prime_detection_count`: Number of primes detected by language
- `graph_f1_score`: Graph-F1 score distribution
- `cultural_adaptations_total`: Cultural adaptations applied

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin):

- **Translation Overview**: Request rates, success rates, error rates
- **Performance Metrics**: Response times, throughput, resource usage
- **Quality Metrics**: Graph-F1 scores, prime detection rates
- **Language Analysis**: Performance by language pair

### Health Checks

- **API Health**: `GET /health`
- **Component Health**: Individual component status
- **Performance Metrics**: `GET /metrics/performance`

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_production_pipeline.py -v
pytest tests/test_api_endpoints.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Demo

```bash
# Run comprehensive demo
python demo/production_demo.py

# Run smoke tests
python smoke_test_suite.py
```

## 🔒 Security

### Production Security Features

- **Non-root containers**: All services run as non-root users
- **Network isolation**: Services communicate via internal network
- **Secrets management**: Environment variables for sensitive data
- **Health checks**: Automatic failure detection and recovery
- **Rate limiting**: Built-in request throttling
- **Input validation**: Comprehensive request validation

### Security Best Practices

1. **Use HTTPS**: Configure SSL/TLS certificates
2. **Update regularly**: Keep dependencies and base images updated
3. **Monitor logs**: Review security-related logs regularly
4. **Access control**: Implement proper authentication/authorization
5. **Backup data**: Regular backups of persistent data

## 📈 Performance

### Performance Characteristics

- **Throughput**: 100+ requests/second (depending on complexity)
- **Latency**: 1-5 seconds per translation (depending on mode)
- **Memory**: 2-4GB RAM usage
- **CPU**: 2-4 cores recommended
- **Storage**: 10-20GB for models and data

### Optimization Tips

1. **Use appropriate modes**: Standard for speed, Neural for quality
2. **Batch processing**: Use batch endpoints for multiple translations
3. **Caching**: Leverage Redis for repeated translations
4. **Resource scaling**: Adjust worker count based on load
5. **Monitoring**: Use metrics to identify bottlenecks

## 🐛 Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   docker-compose logs universal-translator-api
   
   # Check health
   curl http://localhost:8000/health
   ```

2. **High memory usage**
   ```bash
   # Check resource usage
   docker stats
   
   # Restart with more memory
   docker-compose down && docker-compose up -d
   ```

3. **Translation failures**
   ```bash
   # Check API logs
   docker-compose logs -f universal-translator-api
   
   # Test with simple request
   curl -X POST http://localhost:8000/translate \
     -H "Content-Type: application/json" \
     -d '{"source_text":"Hello","source_language":"en","target_language":"es"}'
   ```

### Log Locations

- **Application logs**: `logs/universal-translator.log`
- **Docker logs**: `docker-compose logs <service-name>`
- **Grafana logs**: `docker-compose logs grafana`
- **Prometheus logs**: `docker-compose logs prometheus`

## 🔄 Updates & Maintenance

### Updating the System

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose -f deployment/docker-compose.yml down
docker-compose -f deployment/docker-compose.yml build --no-cache
docker-compose -f deployment/docker-compose.yml up -d
```

### Backup & Restore

```bash
# Backup data
docker-compose -f deployment/docker-compose.yml exec postgres pg_dump -U translator_user universal_translator > backup.sql

# Restore data
docker-compose -f deployment/docker-compose.yml exec -T postgres psql -U translator_user universal_translator < backup.sql
```

## 📚 Documentation

- **API Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Grafana Dashboards**: `http://localhost:3000`
- **Prometheus**: `http://localhost:9090`
- **Kibana**: `http://localhost:5601`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the docs directory
- **Community**: Join our discussion forum

---

**Universal Translator** - Building bridges between languages through semantic understanding.
