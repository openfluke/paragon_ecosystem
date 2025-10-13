# paragon_mnist_service/__init__.py
"""
Paragon MNIST Service
=====================
A FastAPI microservice that:
  • Hosts MNIST sample images (0–9) for cross-service validation
  • Runs the same Paragon model on CPU and GPU backends
  • Provides JSON inference endpoints via /predict and /images/*
"""

__version__ = "0.1.0"
__author__ = "OpenFluke"
__license__ = "Apache-2.0"

from .server import app, main  # so users can run or import directly
