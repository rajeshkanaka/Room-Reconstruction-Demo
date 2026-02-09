"""Room Reconstruction Demo modules package.

Heavy dependencies are imported lazily so lightweight utilities (calibration,
profiles, QA report builders) can be used even when CV/ML dependencies are
not installed yet.
"""

from .calibration import CalibrationInput
from .compliance_profile import ComplianceProfile, get_compliance_profile
from .dxf_exporter import DXFExporter
from .qa_report import build_qa_report, save_qa_report
from .quality_gate import QualityGateResult, evaluate_capture_quality

try:  # Optional heavy imports
    from .depth_estimator import DepthEstimator
    from .dense_reconstructor import DenseReconstructor
    from .floor_plan_generator import FloorPlanGenerator
    from .room_reconstructor import RoomReconstructor
    from .sfm_processor import SfMProcessor
    from .visualizer_3d import ScaleEstimator, Visualizer3D
except Exception:  # pragma: no cover
    pass

__all__ = [
    "CalibrationInput",
    "ComplianceProfile",
    "get_compliance_profile",
    "DXFExporter",
    "build_qa_report",
    "save_qa_report",
    "QualityGateResult",
    "evaluate_capture_quality",
    "DepthEstimator",
    "DenseReconstructor",
    "FloorPlanGenerator",
    "RoomReconstructor",
    "SfMProcessor",
    "ScaleEstimator",
    "Visualizer3D",
]
