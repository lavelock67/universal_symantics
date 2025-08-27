#!/usr/bin/env python3
"""
Demo UI for NSM Universal Translator

A simple web interface to demonstrate the capabilities of our
universal translator stack.
"""

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import requests
import json
import time
from typing import Dict, Any

app = FastAPI(title="NSM Universal Translator Demo", version="1.0.0")

# Templates
templates = Jinja2Templates(directory="demo/templates")

# API base URL
API_BASE_URL = "http://localhost:8001"

@app.get("/", response_class=HTMLResponse)
async def demo_home(request: Request):
    """Demo home page."""
    return templates.TemplateResponse("demo_home.html", {"request": request})

@app.get("/enhanced", response_class=HTMLResponse)
async def enhanced_demo(request: Request):
    """Enhanced demo page."""
    return templates.TemplateResponse("enhanced_demo.html", {"request": request})

@app.get("/showcase", response_class=HTMLResponse)
async def showcase_demo(request: Request):
    """Interactive showcase demo page."""
    return templates.TemplateResponse("showcase_demo.html", {"request": request})

@app.get("/detection", response_class=HTMLResponse)
async def detection_demo(request: Request):
    """Detection demo page."""
    return templates.TemplateResponse("detection_demo.html", {"request": request})

@app.get("/roundtrip", response_class=HTMLResponse)
async def roundtrip_demo(request: Request):
    """Round-trip translation demo."""
    return templates.TemplateResponse("roundtrip_demo.html", {"request": request})

@app.get("/deepnsm", response_class=HTMLResponse)
async def deepnsm_demo(request: Request):
    """DeepNSM demo page."""
    return templates.TemplateResponse("deepnsm_demo.html", {"request": request})

@app.get("/mdl", response_class=HTMLResponse)
async def mdl_demo(request: Request):
    """MDL demo page."""
    return templates.TemplateResponse("mdl_demo.html", {"request": request})

@app.get("/temporal", response_class=HTMLResponse)
async def temporal_demo(request: Request):
    """Temporal reasoning demo page."""
    return templates.TemplateResponse("temporal_demo.html", {"request": request})

@app.get("/ablation", response_class=HTMLResponse)
async def ablation_demo(request: Request):
    """Constraint ablation demo."""
    return templates.TemplateResponse("ablation_demo.html", {"request": request})

@app.get("/exponents", response_class=HTMLResponse)
async def exponents_demo(request: Request):
    """Cross-language exponents demo."""
    return templates.TemplateResponse("exponents_demo.html", {"request": request})

@app.get("/mwe", response_class=HTMLResponse)
async def mwe_demo(request: Request):
    """MWE detection demo."""
    return templates.TemplateResponse("mwe_demo.html", {"request": request})

@app.get("/router", response_class=HTMLResponse)
async def router_demo(request: Request):
    """Risk-coverage router demo."""
    return templates.TemplateResponse("router_demo.html", {"request": request})

@app.get("/discovery", response_class=HTMLResponse)
async def discovery_demo(request: Request):
    """Prime discovery demo."""
    return templates.TemplateResponse("discovery_demo.html", {"request": request})

@app.get("/wow", response_class=HTMLResponse)
async def wow_factor_demo(request: Request):
    """NSM Wow Factor demo."""
    return templates.TemplateResponse("wow_factor_demo.html", {"request": request})

@app.get("/research", response_class=HTMLResponse)
async def research_showcase(request: Request):
    """Advanced research showcase demo."""
    return templates.TemplateResponse("research_showcase.html", {"request": request})

# API proxy endpoints
@app.post("/api/detect")
async def api_detect(request: Request):
    """Proxy to detection API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/detect", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/deepnsm")
async def api_deepnsm(request: Request):
    """Proxy to DeepNSM API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/deepnsm", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/mdl")
async def api_mdl(request: Request):
    """Proxy to MDL API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/mdl", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/temporal")
async def api_temporal(request: Request):
    """Proxy to temporal API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/temporal", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/health")
async def api_health():
    """Proxy to health API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/primes")
async def api_primes():
    """Proxy to primes API."""
    try:
        response = requests.get(f"{API_BASE_URL}/primes")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/mwe")
async def api_mwe(request: Request):
    """Proxy to MWE detection API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/mwe", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/router")
async def api_router(request: Request):
    """Proxy to risk-coverage router API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/router", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/discovery")
async def api_discovery(request: Request):
    """Proxy to prime discovery API."""
    try:
        body = await request.json()
        response = requests.post(f"{API_BASE_URL}/discovery", json=body)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
