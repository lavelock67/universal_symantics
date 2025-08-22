"""Demo API for the Periodic Table of Information Primitives.

This module provides a simple demonstration of the system's capabilities
for cross-modal integration and Φ computation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..table import PeriodicTable, Primitive, PrimitiveCategory, PrimitiveSignature
from ..specialists import TemporalESNSpecialist, CentralHub
from ..validate import CompressionValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Periodic Table of Information Primitives",
    description="Demo API for cross-modal integration and Φ computation",
    version="0.1.0"
)

# Global system components
periodic_table: Optional[PeriodicTable] = None
central_hub: Optional[CentralHub] = None
temporal_specialist: Optional[TemporalESNSpecialist] = None


class SignalInput(BaseModel):
    """Input model for signals."""
    modality: str
    data: List[float]
    metadata: Optional[Dict[str, Any]] = None


class IntegrationRequest(BaseModel):
    """Request model for integration."""
    signals: List[SignalInput]
    compute_phi: bool = True
    temporal_processing: bool = True


class IntegrationResponse(BaseModel):
    """Response model for integration."""
    integration_score: float
    phi_score: float
    attention_weights: Dict[str, float]
    conflicts: List[Dict[str, Any]]
    cross_modal_connections: Dict[str, float]
    temporal_result: Optional[Dict[str, Any]] = None
    workspace_summary: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global periodic_table, central_hub, temporal_specialist
    
    logger.info("Initializing Periodic Primitives System...")
    
    # Create a sample periodic table
    periodic_table = create_sample_periodic_table()
    
    # Initialize temporal specialist
    temporal_specialist = TemporalESNSpecialist(reservoir_size=256, n_esns=3)
    
    # Initialize central hub
    central_hub = CentralHub(periodic_table, temporal_specialist)
    
    logger.info("System initialized successfully")


def create_sample_periodic_table() -> PeriodicTable:
    """Create a sample periodic table with basic primitives."""
    table = PeriodicTable()
    
    # Add some basic primitives
    primitives = [
        # Informational primitives
        Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2),
                 description="Is-a relationship", examples=["cat is animal"]),
        Primitive("SimilarTo", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2),
                 description="Similarity relationship", examples=["cat similar to dog"]),
        
        # Structural primitives
        Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2),
                 description="Part-whole relationship", examples=["wheel part of car"]),
        Primitive("Contains", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2),
                 description="Containment relationship", examples=["box contains items"]),
        
        # Temporal primitives
        Primitive("Before", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2),
                 description="Temporal precedence", examples=["morning before afternoon"]),
        Primitive("During", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2),
                 description="Temporal overlap", examples=["rain during storm"]),
        
        # Causal primitives
        Primitive("Causes", PrimitiveCategory.CAUSAL, PrimitiveSignature(arity=2),
                 description="Causal relationship", examples=["heat causes expansion"]),
        Primitive("Enables", PrimitiveCategory.CAUSAL, PrimitiveSignature(arity=2),
                 description="Enabling relationship", examples=["key enables access"]),
        
        # Spatial primitives
        Primitive("Above", PrimitiveCategory.SPATIAL, PrimitiveSignature(arity=2),
                 description="Spatial relationship", examples=["cloud above ground"]),
        Primitive("Near", PrimitiveCategory.SPATIAL, PrimitiveSignature(arity=2),
                 description="Proximity relationship", examples=["house near park"]),
    ]
    
    for primitive in primitives:
        table.add_primitive(primitive)
    
    logger.info(f"Created periodic table with {len(primitives)} primitives")
    return table


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Periodic Table of Information Primitives Demo API",
        "version": "0.1.0",
        "status": "running",
        "primitives_count": len(periodic_table.primitives) if periodic_table else 0,
        "categories": [cat.value for cat in periodic_table.categories] if periodic_table else []
    }


@app.get("/primitives")
async def get_primitives():
    """Get all primitives in the periodic table."""
    if not periodic_table:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    primitives = []
    for primitive in periodic_table.primitives.values():
        primitives.append({
            "name": primitive.name,
            "category": primitive.category.value,
            "arity": primitive.arity,
            "description": primitive.description,
            "examples": primitive.examples
        })
    
    return {
        "primitives": primitives,
        "total_count": len(primitives),
        "categories": [cat.value for cat in periodic_table.categories]
    }


@app.get("/primitives/{category}")
async def get_primitives_by_category(category: str):
    """Get primitives by category."""
    if not periodic_table:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        category_enum = PrimitiveCategory(category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    
    primitives = periodic_table.get_primitives_by_category(category_enum)
    
    return {
        "category": category,
        "primitives": [
            {
                "name": p.name,
                "arity": p.arity,
                "description": p.description,
                "examples": p.examples
            }
            for p in primitives
        ],
        "count": len(primitives)
    }


@app.post("/integrate", response_model=IntegrationResponse)
async def integrate_signals(request: IntegrationRequest):
    """Integrate signals from multiple modalities."""
    if not central_hub:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if len(request.signals) < 1:
        raise HTTPException(status_code=400, detail="At least one signal required")
    
    # Convert signals to numpy arrays
    signals = {}
    for signal_input in request.signals:
        signals[signal_input.modality] = np.array(signal_input.data)
    
    # Perform integration
    result = central_hub.integrate(signals)
    
    # Prepare response
    response = IntegrationResponse(
        integration_score=result["integration_score"],
        phi_score=result["phi_score"],
        attention_weights=result["attention_weights"],
        conflicts=result["conflicts"],
        cross_modal_connections={
            f"{mod1}-{mod2}": strength 
            for (mod1, mod2), strength in result["cross_modal_connections"].items()
        },
        temporal_result=result.get("temporal_result"),
        workspace_summary=result["workspace_state"]
    )
    
    return response


@app.get("/metrics")
async def get_integration_metrics():
    """Get current integration metrics."""
    if not central_hub:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    metrics = central_hub.get_integration_metrics()
    
    return {
        "current_phi": metrics["current_phi"],
        "current_integration": metrics["current_integration"],
        "phi_history_length": len(metrics["phi_history"]),
        "integration_history_length": len(metrics["integration_history"]),
        "attention_weights": metrics["attention_weights"],
        "cross_modal_connections": {
            f"{mod1}-{mod2}": strength 
            for (mod1, mod2), strength in metrics["cross_modal_connections"].items()
        },
        "conflict_count": metrics["conflict_count"],
        "workspace_modalities": metrics["workspace_modalities"]
    }


@app.post("/temporal/process")
async def process_temporal_sequence(sequence: List[Any]):
    """Process a temporal sequence through the ESN specialist."""
    if not temporal_specialist:
        raise HTTPException(status_code=500, detail="Temporal specialist not initialized")
    
    result = temporal_specialist.process_sequence(sequence)
    
    return {
        "sequence_length": result["sequence_length"],
        "memory_buffer_size": result["memory_buffer_size"],
        "temporal_features_shape": result["temporal_features"].shape if result["temporal_features"].size > 0 else None
    }


@app.get("/temporal/coherence")
async def compute_temporal_coherence(sequence: List[Any]):
    """Compute temporal coherence of a sequence."""
    if not temporal_specialist:
        raise HTTPException(status_code=500, detail="Temporal specialist not initialized")
    
    coherence = temporal_specialist.compute_temporal_coherence(sequence)
    
    return {
        "sequence_length": len(sequence),
        "coherence_score": coherence,
        "interpretation": "high" if coherence > 0.7 else "medium" if coherence > 0.3 else "low"
    }


@app.get("/temporal/memory")
async def get_temporal_memory():
    """Get temporal memory summary."""
    if not temporal_specialist:
        raise HTTPException(status_code=500, detail="Temporal specialist not initialized")
    
    return temporal_specialist.get_memory_summary()


@app.post("/validate/compression")
async def validate_compression(domains: List[str] = ["text", "vision", "logic"]):
    """Run compression validation on specified domains."""
    if not periodic_table:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    validator = CompressionValidator(periodic_table)
    results = validator.test_multiple_domains(domains, n_samples=50)
    
    # Check gates
    gates = validator.check_compression_gates(results, target_ratio=2.0)
    passed_gates = sum(gates.values())
    
    return {
        "results": {
            domain: {
                "avg_compression_ratio": result.get("avg_compression_ratio", 0.0),
                "avg_mdl_score": result.get("avg_mdl_score", 0.0),
                "gate_passed": gates.get(domain, False)
            }
            for domain, result in results.items()
        },
        "gates_passed": passed_gates,
        "total_domains": len(domains),
        "overall_success": passed_gates >= 3
    }


@app.post("/reset")
async def reset_system():
    """Reset the system state."""
    global central_hub, temporal_specialist
    
    if central_hub:
        central_hub.reset()
    if temporal_specialist:
        temporal_specialist.reset()
    
    return {"message": "System reset successfully"}


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Periodic Primitives Demo API...")
    print("Visit http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
