#!/usr/bin/env python3
"""Test FastAPI MWE serialization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.core.domain.models import MWE, MWEType, Language
from pydantic import BaseModel
from typing import List

# Create a simple test app
app = FastAPI()

class TestMWEResponse(BaseModel):
    mwes: List[MWE]

@app.get("/test")
def test_mwe():
    """Test MWE serialization."""
    mwe = MWE(
        text="at least",
        type=MWEType.QUANTIFIER,
        primes=["MORE"],
        confidence=0.8,
        start=0,
        end=8,
        language=Language.ENGLISH
    )
    
    return TestMWEResponse(mwes=[mwe])

def test_fastapi_serialization():
    """Test FastAPI serialization."""
    client = TestClient(app)
    response = client.get("/test")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        data = response.json()
        mwe = data["mwes"][0]
        print(f"MWE primes: {mwe.get('primes', 'NOT FOUND')}")

if __name__ == "__main__":
    test_fastapi_serialization()
