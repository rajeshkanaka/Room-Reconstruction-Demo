# Room Reconstruction Pipeline
## From Photos to Floor Plans

**How 4-24 photos become a 2D floor plan with measurements.**

---

## Pipeline Flow

```
Input Photos (4-24)
        ↓
1. Load Images (OpenCV, PIL, NumPy)
        ↓
       ┌─┴─┐
       ▼   ▼
2A. VGGT (3D)    2B. Gemini (Room Info)
   • Poses           • Room type
   • Depth           • Door count
   • Point cloud     • Window count
   (PyTorch)        (Google AI)
       └─┬─┘
         ↓
3. Combine Data (NumPy)
        ↓
4. Find Floor Plane (Open3D)
        ↓
5. Extract 2D Points (NumPy)
        ↓
6. Create Density Map (NumPy)
        ↓
7. Clean Map (OpenCV, SciPy)
        ↓
8. Detect Walls (OpenCV, SciPy)
        ↓
9. Find Room Boundary (OpenCV, Shapely)
        ↓
10. Calculate Size (NumPy)
        ↓
11. Quality Check
        ↓
12. Create Floor Plans
    ├─ PNG (Matplotlib)
    ├─ SVG (svgwrite)
    └─ DXF (ezdxf)
        ↓
13. 3D Visualization (Plotly, Open3D)
        ↓
OUTPUT: Floor plans + 3D Model + Measurements
```

---

## Step Details

### Step 1: Load Images
**Library:** OpenCV, PIL, NumPy
**What:** Read photos, convert to RGB

### Step 2A: VGGT (Primary AI)
**Library:** VGGT Model, PyTorch
**What:** Process all photos once. Find camera positions, depth maps, 3D point cloud.
**Key:** Single forward pass - fast and accurate.

### Step 2B: Gemini 3
**Library:** Google Gemini AI
**What:** Runs in parallel (no extra time). Analyzes room type, doors, windows.

### Step 3: Combine Data
**Library:** NumPy
**What:** Merge VGGT's 3D points with Gemini's room details.

### Step 4: Find Floor Plane
**Library:** Open3D
**What:** Detect floor surface (RANSAC), keep floor points.

### Step 5: Extract 2D Points
**Library:** NumPy
**What:** Take floor slice (10-30 cm), project to top-down view.

### Step 6: Create Density Map
**Library:** NumPy
**What:** Create 100x100 grid, count points per cell.

### Step 7: Clean the Map
**Library:** OpenCV, SciPy
**What:** Threshold, morphological ops, fill holes.

### Step 8: Detect Walls
**Library:** OpenCV, SciPy
**What:** Hough transform for lines, edge detection.

### Step 9: Find Room Boundary
**Library:** OpenCV, Shapely
**What:** Trace contours, simplify, close polygon.

### Step 10: Calculate Size
**Library:** NumPy
**What:** Calculate width, depth, area from 3D coordinates.

### Step 11: Quality Check
**What:** Score 0-1. Export: Normal / Annotate / More images.

### Step 12: Create Floor Plans
**PNG (Matplotlib):** Architectural drawing with measurements.
**SVG (svgwrite):** Vector - zoom infinitely.
**DXF (ezdxf):** CAD file for AutoCAD, FreeCAD.

### Step 13: 3D Visualization
**Library:** Plotly, Open3D
**What:** Interactive HTML, PLY point cloud.

---

## Summary Flow

```
Photos → VGGT (3D) + Gemini (Info) → Combine → Floor
→ 2D Grid → Clean → Walls → Boundary → Size → Quality
→ PNG/SVG/DXF + 3D Model
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| **VGGT** | 3D reconstruction AI |
| **Google Gemini** | Room understanding |
| **PyTorch** | AI model running |
| **OpenCV** | Image processing |
| **NumPy** | Math and arrays |
| **Open3D** | 3D operations |
| **SciPy** | Scientific computing |
| **Shapely** | Geometry |
| **Matplotlib** | PNG rendering |
| **svgwrite** | SVG files |
| **ezdxf** | DXF CAD files |
| **Plotly** | Interactive 3D |

---

## Output Files (in `./outputs/`)

- `floor_plan_arch_*.png` - Architectural image
- `floor_plan_*.svg` - Vector floor plan
- `floor_plan_*.dxf` - CAD file
- `room_3d_*.html` - Interactive 3D view
- `room_pointcloud_*.ply` - 3D point cloud

---

## Performance

- **VGGT model:** ~4GB (first time only)
- **Processing time:** 30-60 seconds for 4-24 photos
- **Gemini:** Runs in parallel (no extra time)

---

## Accuracy

- **VGGT path:** Metric accuracy (actual meters)
- **Quality score:** 0-1, higher is better
- **Industry ready:** Only if score > 0.7
