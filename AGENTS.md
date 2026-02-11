# AGENTS.md

This file guides AI coding agents working on this repository.

## Essential Commands

```bash
# Installation
uv pip install -e ../vggt          # VGGT from local clone
uv pip install -r requirements.txt  # All dependencies

# Running
uv run python app.py               # Gradio web UI (http://localhost:7850)
uv run python run_cli.py photos/*.jpg --room-width 5.0 --visualize

# Testing
pytest                              # All tests (excludes slow by default)
pytest -m slow                      # Only slow/integration tests
pytest -m "not slow"               # Exclude slow tests
pytest tests/test_phase1_metric_depth.py              # Single test file
pytest tests/test_e2e.py::TestFullPipeline::test_floor_plan_model_full_lifecycle  # Specific test
```

## Code Style Guidelines

### Imports
Order: standard library → third-party → local. Group with blank lines.

```python
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from config import ENABLE_VGGT, VGGT_MODEL
from modules.vggt_reconstructor import VGGTReconstructor
```

### Naming Conventions
- **Classes:** PascalCase (`VGGTReconstructor`, `FloorPlanGenerator`)
- **Functions/Methods:** snake_case (`reconstruct`, `generate_floor_plan`)
- **Constants:** UPPER_CASE in `config.py` (`ENABLE_VGGT`, `VGGT_MODEL`)
- **Private methods:** underscore prefix (`_load_model`, `_preprocess`)

### Type Hints & Docstrings
Use type hints for function signatures. Triple-quoted docstrings for modules/classes/methods.

```python
def reconstruct(self, images: list[np.ndarray]) -> dict:
    """Single-pass reconstruction from RGB images."""
    pass

class SceneAnalysis:
    success: bool = False
    room_type: str = ""
    confidence: float = 0.0
```

### Error Handling
Use try/except with descriptive messages. Fail gracefully.

```python
try:
    result = self.model(input_tensor)
    return {"success": True, "data": result}
except Exception as e:
    print(colored(f"[Module] Operation failed: {e}", "yellow"))
    return {"success": False, "error": str(e)}
```

### Logging
Use `termcolor.colored` for colored terminal output with module prefixes.

```python
from termcolor import colored
print(colored("[VGGT] Processing 5 images...", "cyan"))
print(colored(f"[VGGT] Complete: {n:,} points", "green"))
print(colored("[VGGT] Warning: low confidence", "yellow"))
```

### Path Handling
Use `os.path` for cross-platform paths. Use `sys.path.insert` in tests.

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# In tests:
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Configuration
All parameters live in `config.py`. Import from there, never hardcode.

```python
from config import ENABLE_VGGT, VGGT_MODEL, VGGT_CONFIDENCE_THRESHOLD
```

### Test Organization
- Test files: `tests/test_phase1_*.py`, `tests/test_e2e.py`
- Mark slow tests with `@pytest.mark.slow`
- Use synthetic data where possible to avoid model loading

```python
import pytest

class TestVGGT:
    @pytest.mark.slow
    def test_estimate_depth_shape(self):
        from modules.vggt_reconstructor import VGGTReconstructor
```

### Data Structures
Use `@dataclass` for structured return types.

```python
from dataclasses import dataclass, field

@dataclass
class SceneAnalysis:
    success: bool = False
    room_type: str = ""
    doors: list = field(default_factory=list)
```

### NumPy Patterns
Prefer vectorized operations. Handle NaN/Inf checks.

```python
valid_mask = np.isfinite(points).all(axis=1)
points = points[valid_mask]

# Percentiles for robust bounds
x_min, x_max = np.percentile(x_coords, [1, 99])
```

### Backend Switching
Primary: VGGT (`ENABLE_VGGT=True`). Fallback: legacy SfM. Check config flags.

```python
if ENABLE_VGGT:
    from modules.vggt_reconstructor import VGGTReconstructor
    reconstructor = VGGTReconstructor()
else:
    # Legacy path
```

### Output Files
All outputs to `./outputs/`. Create directory if needed.

```python
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "floor_plan.png")
```

## Project Architecture

**Primary:** Photos → VGGT (metric depth + poses + points) ‖ Gemini 3 (semantic) → Wall detection → Floor plan + 3D

**Core Modules:**
- `modules/room_reconstructor.py` - Orchestrator
- `modules/vggt_reconstructor.py` - VGGT backend (primary)
- `modules/scene_analyzer.py` - Gemini 3 semantic analysis
- `modules/floor_plan_generator.py` - Floor plan generation
- `modules/geometry/floor_plan_model.py` - Data model
- `modules/rendering/` - SVG/DXF/PNG renderers

**Entry Points:**
- `app.py` - Gradio web UI
- `run_cli.py` - Command-line interface

## Key Design Decisions

- VGGT is primary backend (single forward pass, metric output)
- Gemini runs in parallel via ThreadPoolExecutor (zero latency)
- Graceful degradation: VGGT unavailable → legacy SfM pipeline
- Config-driven feature flags: `ENABLE_VGGT`, `ENABLE_GEMINI_ANALYSIS`
- All parameters in `config.py` for easy tuning
