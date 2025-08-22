# Periodic Table of Information Primitives

A systematic approach to discovering and organizing fundamental information primitives across multiple modalities (text, vision, logic) through cross-model mining, validation, and integration.


## Overview

This project implements a "periodic table" of information primitives by:
- **Mining** stable patterns across diverse embedding models and knowledge graphs
- **Validating** primitives through compression, reconstruction, and transfer tests
- **Integrating** cross-modal information using difference-based memory and temporal coherence
- **Measuring** integration (Φ) as a proxy for consciousness-like properties

## Architecture

```
periodic-primitives/
├── src/
│   ├── mining/          # Primitive discovery from embeddings, KGs, vision
│   ├── table/           # Periodic table schema and algebra
│   ├── validate/        # Compression, reconstruction, transfer tests
│   ├── specialists/     # Domain-specific processors + integration hub
│   └── ui/             # Demo API and visualization
├── data/               # ConceptNet, samples, indices
└── tests/              # Comprehensive test suite
```

## Quick Start

1. **Setup Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

2. **Run Mining Pipeline**
   ```bash
   python -m src.mining.embedding_miner --models bert gpt2 --concepts 500
   python -m src.mining.conceptnet_miner --languages 20
   ```

3. **Validate Primitives**
   ```bash
   python -m src.validate.compression --domains text vision logic
   ```

4. **Demo Integration**
   ```bash
   python -m src.ui.demo_api
   ```

## Core Components

### Primitive Mining
- **Embedding Intersection**: Cross-model stable directions via Procrustes + NMF
- **KG Universals**: Multilingual ConceptNet relations present across ≥20 languages
- **Vision Bases**: Conv filter clustering from multiple vision backbones
- **Logic/Math**: Operator/quantifier patterns from code/math corpora

### Validation Battery
- **Compression**: >2× compression vs. naive baselines
- **Reconstruction**: <10% error across domains
- **Transfer**: Positive cross-domain transfer learning
- **Integration**: Φ proxy for coherent multi-modal inputs

### Specialists & Integration
- **Domain Specialists**: Vision, language, logic processors
- **Temporal ESN**: Echo-state networks for sequence memory
- **Central Hub**: Cross-modal integration with Φ computation

## Success Metrics

- **Universality**: Jaccard overlap >0.5 across models
- **Parsimony**: 100-200 primitives total
- **Compression**: >2× across 3 domains
- **Transfer**: ≥+5pp vs. baseline on cross-domain tasks
- **Integration**: Φ significantly higher for coherent inputs

## Development

This project follows Python best practices:
- Type hints throughout
- PEP 257 docstrings
- Modular classes and functions
- Comprehensive error handling
- Incremental development with git branches

## License

MIT License - see LICENSE file for details.

## Contributing

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make incremental commits with clear messages
3. Add tests for new functionality
4. Ensure all validation gates pass
5. Submit a pull request

## Roadmap

- **Week 1-2**: Primitive mining (embedding intersection, KG universals, vision bases)
- **Week 3**: Table induction & algebra implementation
- **Week 4**: Validation battery (compression, reconstruction, transfer)
- **Week 5-6**: Specialists + integration hub
- **Week 7**: Structured complexity upgrade (height links)
- **Week 8**: MVP demo & documentation
