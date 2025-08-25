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

@app.get("/roundtrip", response_class=HTMLResponse)
async def roundtrip_demo(request: Request):
    """Round-trip translation demo."""
    return templates.TemplateResponse("roundtrip_demo.html", {"request": request})

@app.post("/roundtrip", response_class=HTMLResponse)
async def roundtrip_demo_post(request: Request, 
                             source_text: str = Form(...),
                             src_lang: str = Form("en"),
                             tgt_lang: str = Form("es"),
                             constraint_mode: str = Form("hybrid")):
    """Process round-trip translation demo."""
    
    try:
        # Call the API
        response = requests.post(
            f"{API_BASE_URL}/roundtrip",
            json={
                "source_text": source_text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "constraint_mode": constraint_mode,
                "realizer": "fluent"
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "roundtrip_result.html", 
                {
                    "request": request,
                    "input_text": source_text,
                    "result": result,
                    "success": True
                }
            )
        else:
            return templates.TemplateResponse(
                "roundtrip_result.html",
                {
                    "request": request,
                    "input_text": source_text,
                    "error": f"API Error: {response.status_code}",
                    "success": False
                }
            )
            
    except Exception as e:
        return templates.TemplateResponse(
            "roundtrip_result.html",
            {
                "request": request,
                "input_text": source_text,
                "error": str(e),
                "success": False
            }
        )

@app.get("/mwe", response_class=HTMLResponse)
async def mwe_demo(request: Request):
    """MWE detection demo."""
    return templates.TemplateResponse("mwe_demo.html", {"request": request})

@app.post("/mwe", response_class=HTMLResponse)
async def mwe_demo_post(request: Request, text: str = Form(...)):
    """Process MWE detection demo."""
    
    try:
        # Call the API
        response = requests.post(
            f"{API_BASE_URL}/mwe",
            json={
                "text": text,
                "include_coverage": True
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "mwe_result.html",
                {
                    "request": request,
                    "input_text": text,
                    "result": result,
                    "success": True
                }
            )
        else:
            return templates.TemplateResponse(
                "mwe_result.html",
                {
                    "request": request,
                    "input_text": text,
                    "error": f"API Error: {response.status_code}",
                    "success": False
                }
            )
            
    except Exception as e:
        return templates.TemplateResponse(
            "mwe_result.html",
            {
                "request": request,
                "input_text": text,
                "error": str(e),
                "success": False
            }
        )

@app.get("/router", response_class=HTMLResponse)
async def router_demo(request: Request):
    """Risk-coverage router demo."""
    return templates.TemplateResponse("router_demo.html", {"request": request})

@app.post("/router", response_class=HTMLResponse)
async def router_demo_post(request: Request, text: str = Form(...), lang: str = Form("en")):
    """Process risk-coverage router demo."""
    
    try:
        # Call the API
        response = requests.post(
            f"{API_BASE_URL}/router/route",
            json={
                "text": text,
                "lang": lang
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "router_result.html",
                {
                    "request": request,
                    "input_text": text,
                    "result": result,
                    "success": True
                }
            )
        else:
            return templates.TemplateResponse(
                "router_result.html",
                {
                    "request": request,
                    "input_text": text,
                    "error": f"API Error: {response.status_code}",
                    "success": False
                }
            )
            
    except Exception as e:
        return templates.TemplateResponse(
            "router_result.html",
            {
                "request": request,
                "input_text": text,
                "error": str(e),
                "success": False
            }
        )

@app.get("/ablation", response_class=HTMLResponse)
async def ablation_demo(request: Request):
    """Constraint ablation demo."""
    return templates.TemplateResponse("ablation_demo.html", {"request": request})

@app.post("/ablation", response_class=HTMLResponse)
async def ablation_demo_post(request: Request, text: str = Form(...), lang: str = Form("en")):
    """Process constraint ablation demo."""
    
    try:
        # Call the API
        response = requests.post(
            f"{API_BASE_URL}/ablation",
            json={
                "text": text,
                "lang": lang,
                "modes": ["off", "hybrid", "hard"]
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "ablation_result.html",
                {
                    "request": request,
                    "input_text": text,
                    "result": result,
                    "success": True
                }
            )
        else:
            return templates.TemplateResponse(
                "ablation_result.html",
                {
                    "request": request,
                    "input_text": text,
                    "error": f"API Error: {response.status_code}",
                    "success": False
                }
            )
            
    except Exception as e:
        return templates.TemplateResponse(
            "ablation_result.html",
            {
                "request": request,
                "input_text": text,
                "error": str(e),
                "success": False
            }
        )

@app.get("/exponents", response_class=HTMLResponse)
async def exponents_demo(request: Request):
    """Cross-language exponents demo."""
    return templates.TemplateResponse("exponents_demo.html", {"request": request})

@app.post("/exponents", response_class=HTMLResponse)
async def exponents_demo_post(request: Request, 
                             prime: str = Form(...),
                             language: str = Form("en"),
                             register: str = Form("neutral")):
    """Process cross-language exponents demo."""
    
    try:
        # Call the API
        response = requests.post(
            f"{API_BASE_URL}/exponents",
            json={
                "prime": prime,
                "language": language,
                "register": register
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "exponents_result.html",
                {
                    "request": request,
                    "prime": prime,
                    "language": language,
                    "result": result,
                    "success": True
                }
            )
        else:
            return templates.TemplateResponse(
                "exponents_result.html",
                {
                    "request": request,
                    "prime": prime,
                    "language": language,
                    "error": f"API Error: {response.status_code}",
                    "success": False
                }
            )
            
    except Exception as e:
        return templates.TemplateResponse(
            "exponents_result.html",
            {
                "request": request,
                "prime": prime,
                "language": language,
                "error": str(e),
                "success": False
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
