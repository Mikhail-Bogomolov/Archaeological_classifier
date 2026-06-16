"""Модели и обучение."""

__all__ = ["ArchaeologyClassifierPipeline"]


def __getattr__(name: str):
    if name == "ArchaeologyClassifierPipeline":
        from app.ml.pipeline import ArchaeologyClassifierPipeline
        return ArchaeologyClassifierPipeline
    raise AttributeError(name)
