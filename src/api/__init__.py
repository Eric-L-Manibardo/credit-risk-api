"""HTTP service. The model is loaded once at startup, not per request."""

from src.api.app import app, create_app

__all__ = ["app", "create_app"]
