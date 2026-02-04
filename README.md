# 🏠 Room Reconstruction Demo

**Transform simple room photos into 2D floor plans and interactive 3D models using AI!**

This is a proof-of-concept demonstration application that shows how computer vision and AI can be used to reconstruct room layouts from just a few photographs. Built for the US home renovation market, this demo allows inspectors and property assessors to quickly generate approximate floor plans and 3D visualizations without special equipment.

![Demo Overview](docs/demo_overview.png)

---

## 🎯 What This Demo Does

- **Input:** 4-5 photographs of a room taken from different angles
- **Output:**
  - 2D Floor Plan with approximate dimensions (meters & feet)
  - Interactive 3D Model visualization
  - Room measurements (width, depth, area)
  - Downloadable files (PLY point cloud, HTML 3D viewer)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU with CUDA support (optional but recommended for speed)

### Installation

```bash
# 1. Clone or navigate to the project directory
cd room_reconstruction_demo

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Note: On first run, the AI model (~350MB) will be downloaded automatically
```

### Running the Demo

#### Option 1: Web Interface (Recommended)

```bash
python app.py
```

Then open your browser to: **http://localhost:7860**

#### Option 2: Command Line

```bash
python run_cli.py path/to/image1.jpg path/to/image2.jpg path/to/image3.jpg
```

---

## 📷 How to Take Good Photos

For best results, follow these guidelines when photographing the room:

### Photo Recommendations

1. **Take 4-5 photos** from different corners of the room
2. **Stand in corners** and photograph towards the opposite corner
3. **Include the floor** and at least 2 walls in each shot
4. **Use good lighting** - natural daylight works best
5. **Hold the camera level** - avoid tilting up or down

### Example Photo Positions

```
┌───────────────────┐
│ 📷⇘            📷⇙ │   Photo 1: From corner A looking at C
│  A              B  │   Photo 2: From corner B looking at D
│                    │   Photo 3: From corner C looking at A
│                    │   Photo 4: From corner D looking at B
│  D              C  │   Photo 5: From center (optional)
│ 📷⇗            📷⇖ │
└───────────────────┘
```

### What to Avoid

- ❌ Photos from the same angle
- ❌ Extreme wide-angle distortion
- ❌ Very dark or overexposed images
- ❌ Photos that don't show the floor
- ❌ Motion blur

---

## 📂 Project Structure

```
room_reconstruction_demo/
├── app.py                    # Main Gradio web application
├── run_cli.py                # Command-line interface
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── modules/                  # Core processing modules
│   ├── __init__.py
│   ├── depth_estimator.py    # AI depth estimation from images
│   ├── floor_plan_generator.py  # 2D floor plan creation
│   ├── visualizer_3d.py      # 3D visualization tools
│   └── room_reconstructor.py # Main orchestration
│
├── outputs/                  # Generated outputs (auto-created)
└── sample_images/            # Sample test images (add your own)
```

---

## 🔧 How It Works

This demo uses a multi-stage pipeline to convert 2D photos into 3D representations:

### Stage 1: Depth Estimation

```
🖼️ Photo → [DPT AI Model] → 🗺️ Depth Map
```

We use the **DPT (Dense Prediction Transformer)** model from Intel, pre-trained on millions of images to predict depth. For each pixel in the photo, the AI estimates how far away that point is from the camera.

### Stage 2: 3D Point Cloud Generation

```
🗺️ Depth Map + 📷 Camera Model → ☁️ 3D Point Cloud
```

Using the pinhole camera model and the depth map, we project each pixel into 3D space, creating a colored point cloud.

### Stage 3: Multi-View Combination

```
☁️ Cloud 1 + ☁️ Cloud 2 + ☁️ Cloud 3... → 🏠 Combined Model
```

Point clouds from multiple views are combined (with simple rotation offsets) to build a more complete room representation.

### Stage 4: Floor Plan Generation

```
🏠 3D Model → [Top-Down Projection] → 📏 Floor Plan
```

We take a "bird's eye view" of the point cloud, creating a 2D density map that represents the room layout. Edge detection identifies walls.

### Stage 5: Visualization

```
Floor Plan → 📊 2D Image with Measurements
3D Model → 🎮 Interactive Plotly Viewer
```

---

## ⚙️ Configuration

You can adjust settings in `config.py`:

```python
# Depth estimation
DEPTH_MAX_SIZE = 512      # Max image size (lower = faster)

# 3D reconstruction
POINT_CLOUD_DENSITY = 4   # Sample rate (higher = fewer points, faster)
VOXEL_SIZE = 0.05         # Voxel size for downsampling

# Floor plan
ASSUMED_ROOM_WIDTH_METERS = 4.0  # Default room width assumption
```

---

## 📊 Understanding the Output

### Floor Plan

The floor plan shows:
- **Layout Map:** Top-down view showing point density (blue = more data)
- **Schematic:** Simplified room outline with dimensions
- **Measurements:** Width, depth, and area in both metric and imperial

### 3D Model

The interactive 3D viewer allows you to:
- Rotate the model (click and drag)
- Zoom in/out (scroll)
- Pan (right-click and drag)
- Hover for coordinates

### Exported Files

Outputs are saved to the `outputs/` folder:
- `floor_plan_TIMESTAMP.png` - Floor plan image
- `room_3d_TIMESTAMP.html` - Standalone 3D viewer (shareable)
- `room_pointcloud_TIMESTAMP.ply` - Point cloud file (for 3D software)

---

## ⚠️ Limitations & Caveats

### Accuracy

- **Measurements are APPROXIMATE** - typically within ±20-30%
- The system assumes a room width to calculate scale
- Depth estimation is based on AI inference, not actual measurements
- **Not suitable for construction, legal, or professional purposes**

### Technical Limitations

- Works best with **rectangular rooms**
- Struggles with very cluttered spaces
- Performance depends on image quality and lighting
- Multiple views are combined with simple offsets (no proper registration)

### What This Demo Cannot Do

- Detect doors, windows, or furniture precisely
- Provide sub-centimeter accuracy
- Work reliably with just 1 photo
- Handle outdoor scenes or very large spaces

---

## 🔮 Future Improvements

For a production system, consider:

1. **Better Multi-View Registration:** Use ICP or feature matching to properly align point clouds
2. **Reference Object Scaling:** Detect known objects (doors, A4 paper) for accurate scale
3. **Semantic Segmentation:** Identify walls, floors, furniture, doors, windows
4. **Advanced Reconstruction:** Use NeRF or 3D Gaussian Splatting for photorealistic results
5. **LiDAR Integration:** Support phones with LiDAR for direct depth capture
6. **Cloud Processing:** Offload heavy computation to cloud GPUs

---

## 🛠️ Troubleshooting

### "CUDA out of memory" Error

Reduce memory usage:
```python
# In config.py
DEPTH_MAX_SIZE = 384  # Reduce from 512
POINT_CLOUD_DENSITY = 6  # Increase from 4
```

Or use CPU (slower but works):
```python
# In modules/depth_estimator.py, change:
self.device = "cpu"  # Force CPU
```

### Slow Processing

- Use a GPU if available (install CUDA-enabled PyTorch)
- Reduce image sizes before uploading
- Adjust `POINT_CLOUD_DENSITY` in config

### Poor Results

- Take more photos from different angles
- Ensure good, even lighting
- Avoid extreme lens distortion
- Try adjusting the assumed room width

### Model Download Fails

The DPT model is downloaded from HuggingFace on first run. If it fails:
```bash
# Try manual download
python -c "from transformers import DPTImageProcessor, DPTForDepthEstimation; DPTImageProcessor.from_pretrained('Intel/dpt-large'); DPTForDepthEstimation.from_pretrained('Intel/dpt-large')"
```

---

## 📚 Dependencies

Main libraries used:

| Library | Purpose |
|---------|----------|
| PyTorch | Deep learning framework |
| Transformers | DPT depth estimation model |
| Open3D | 3D point cloud processing |
| OpenCV | Image processing |
| Plotly | Interactive 3D visualization |
| Matplotlib | 2D plotting |
| Gradio | Web interface |

---

## 📄 License

This demo is provided for educational and proof-of-concept purposes.

---

## 🙏 Acknowledgments

- **Intel** for the DPT depth estimation model
- **HuggingFace** for model hosting and Transformers library
- **Hosta.ai** for inspiration on the use case
- **Open3D** team for excellent 3D processing tools

---

*Built as a proof-of-concept for the US home renovation market* 🏠
