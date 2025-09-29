#!/usr/bin/env python3
"""
FastAPI Service Launcher
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from serving.app import app

if __name__ == "__main__":
    uvicorn.run(
        "serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
