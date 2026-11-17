#!/usr/bin/env python3
"""起動スクリプト"""
import uvicorn
from backend.config import load_config

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    uvicorn.run(
        "backend.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        log_level="info",
    )
