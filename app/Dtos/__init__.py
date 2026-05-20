"""
Media worker DTOs.

  TaskDto/
    MediaTask     (+ FrameSource)         orchestrator → worker
    MediaResult   (+ Bbox, RegionCrops,    worker → orchestrator
                     DetectedFace)
"""
from app.Dtos.TaskDto import (
    MediaTask, FrameSource,
    MediaResult, DetectedFace, RegionCrops, Bbox,
)

__all__ = [
    "MediaTask", "FrameSource",
    "MediaResult", "DetectedFace", "RegionCrops", "Bbox",
]