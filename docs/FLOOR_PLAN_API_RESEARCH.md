# Commercial Floor Plan APIs & Services: Research Report

**Date:** 2026-02-09
**Purpose:** Evaluate commercial APIs and services for floor plan generation and room layout estimation that could augment or replace the current custom CV pipeline
**Context:** Current system uses Hough line transforms on depth slices, convex hull boundaries (convex rooms only), and has ~20-30% measurement error

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Detailed Evaluations](#2-detailed-evaluations)
   - 2.1 CubiCasa
   - 2.2 magicplan
   - 2.3 Apple RoomPlan API
   - 2.4 Matterport API
   - 2.5 Planner 5D
   - 2.6 Floorfy / iGUIDE
   - 2.7 LLM Vision (GPT-4V / Claude Vision)
   - 2.8 FloorNet / Floor-SP / PolyRoom / BADGR / FloorSAM (Academic)
   - 2.9 Polycam
   - 2.10 RasterScan (Floor Plan Digitalization API)
   - 2.11 Floorplanner API
   - 2.12 MeasureSquare SDK
   - 2.13 Zillow Indoor Dataset + AI Floor Plans
3. [Comparison Matrix](#3-comparison-matrix)
4. [Ranking & Recommendations](#4-ranking--recommendations)
5. [Implementation Roadmaps (Top 2 Options)](#5-implementation-roadmaps)
6. [Sources](#6-sources)

---

## 1. Executive Summary

### Key Findings

The commercial floor plan API landscape divides into four categories:

| Category | Examples | Relevance to This Project |
|----------|----------|--------------------------|
| **Scan-to-floorplan services** | CubiCasa, magicplan, Polycam | High -- require a scan walkthrough, not static photos |
| **Hardware-dependent platforms** | RoomPlan (LiDAR), Matterport (3D camera), iGUIDE (laser) | Medium -- excellent accuracy but need specific hardware |
| **Image-to-floorplan APIs** | RasterScan, Floorplanner ML, TF2DeepFloorplan | High -- accept image input, closest to current workflow |
| **Academic models** | FloorNet, Floor-SP, PolyRoom, BADGR, FloorSAM | High -- open-source, state-of-the-art algorithms |

### Critical Gap

No commercial API currently accepts 4-5 static room photos as input and returns a measured floor plan with non-convex room support. This is exactly what the current project attempts. The closest alternatives are:

1. **CubiCasa** -- accepts a video walkthrough (not photos) and returns accurate floor plans via REST API. The best commercial option if input requirements are relaxed to accept video.
2. **Academic models (BADGR)** -- accepts wide-baseline RGB panoramas and jointly reconstructs floor plans with bundle adjustment. Open-source, closest to the project's technical approach.
3. **RasterScan** -- accepts floor plan images (blueprints/sketches) and digitizes them. Only useful as a post-processing step if a raster floor plan already exists.

### Recommendation

The two best strategies are:

- **Strategy A (Hybrid Commercial):** Integrate CubiCasa's SDK for scanning + API for processing. Change input from "4-5 photos" to "video walkthrough." Achieves 95-97% accuracy immediately.
- **Strategy B (Academic + Custom):** Integrate BADGR or PolyRoom for floor plan reconstruction, combine with Depth Pro for metric depth, and Shapely for non-convex polygon handling. Preserves the "photos as input" approach but requires significant engineering.

---

## 2. Detailed Evaluations

### 2.1 CubiCasa

**What it does:** Generates professional 2D and 3D floor plans from a smartphone video walkthrough. AI converts the video scan into a floor plan with room dimensions, wall detection, and door/window placement. Floor plans are delivered within 6-24 hours (or 6 hours with expedited add-on).

**API availability:** YES -- full REST API
- **Integrate API** at `https://app.cubi.casa/api/integrate/v3`
- Conversion API for processing uploaded scans
- Mobile SDKs (iOS via Swift Package Manager, Android AAR library)
- Staging environment for development: `https://qa-customers.cubi.casa/api/integrate/v3`
- Developer support at developer.support@cubicasa.com

**Input requirements:**
- Video walkthrough captured via CubiCasa mobile app (or custom app using their SDK)
- Scan takes ~5-10 minutes per property
- No special hardware required (works on any smartphone camera; LiDAR-equipped devices give better results)
- NOT static photos -- requires walking through the space

**Output format:**
- PNG, JPG (raster floor plans)
- SVG (vector)
- DXF (CAD-compatible, available on higher tiers)
- GLA reports (ANSI Z765 compliant)
- Room dimensions, square footage
- 2D and 3D floor plans

**Measurement accuracy:**
- Average accuracy: 95-97%
- LiDAR-equipped devices: within 3%
- Independent testing shows 3"-11" error depending on room (varies by room complexity and scanning conditions)
- Quality-assured by human QA engineers before delivery
- Meets ANSI Z765 standard for GLA reports

**Integration complexity for Python pipeline:**
- Medium. REST API is straightforward to call from Python.
- However, input must come from their mobile SDK -- cannot feed static photos.
- Would need to build a mobile companion app or change the workflow to video capture.
- API is described as "work in progress" with some undocumented features.

**Non-convex room handling:** YES -- CubiCasa handles L-shaped, U-shaped, and complex room geometries since it reconstructs from full video walkthroughs with AI interpretation.

**Commercial licensing and pricing:**
- Free tier (LITE): unlimited free floor plans up to 5,000 sq ft (PNG/JPG only, black & white)
- Paid plans: $22.99-$29.99 per floor plan
- Add-ons: $15 each for furniture details, GLA report, 6-hour turnaround
- Volume discounts at 20+ scans/month
- API access requires paid subscription

**Verdict:** Best commercial option for accuracy and API maturity. The limitation is that it requires video input, not static photos. Redfin has integrated CubiCasa into its home search platform, validating the technology at scale.

---

### 2.2 magicplan

**What it does:** Creates floor plans via room-by-room scanning using AR technology on mobile devices. Captures rooms individually with the device camera (or LiDAR on supported devices), detects doors and windows automatically, and generates dimensioned floor plans.

**API availability:** YES -- REST API v2
- Base URL: `https://cloud.magicplan.app/api/v2`
- Client libraries for Python, Ruby, Node.js, PHP, Shell
- Webhooks for event-driven workflows
- API key + Customer ID authentication (header-based)
- Rate limits: 500 requests/5 min (standard), 2,000 requests/5 min (high-throughput)
- Requires Report or PRO subscription

**Input requirements:**
- Room-by-room scanning via mobile app (AR camera or LiDAR auto-scan)
- Optional: Bluetooth laser distance meters (Hilti, Bosch, DeWalt, Leica, Stabila, Wurth) for exact measurements
- Optional: Ricoh 360 cameras
- NOT static photos

**Output format:**
- PDF (sketch with dimensions)
- Image files (JPEG, PNG -- compatible with AutoCAD workflows)
- 3D export (multiple formats including OBJ for 3D software)
- Statistics export (CSV, PDF) -- surface area, perimeter, living area
- Xactimate ESX integration (direct)

**Measurement accuracy:**
- With Bluetooth laser meter: claimed 100% accuracy
- Camera-only: "surprisingly reliable" per user reviews, minor adjustments needed
- No published percentage without laser
- Door and window detection described as "surprisingly reliable"

**Integration complexity for Python pipeline:**
- Medium. REST API is well-documented with Python client library support.
- Same limitation as CubiCasa: requires mobile scan input, not static photos.
- API is primarily for project management (CRUD on projects), not for submitting raw images for processing.
- Webhooks enable asynchronous processing workflows.

**Non-convex room handling:** YES -- scans rooms individually and stitches them together, so L-shaped and complex layouts are handled naturally by scanning each wing separately.

**Commercial licensing and pricing:**
- Starting at $149/month
- 10 new projects/month included, rollover unused credits
- Extra projects: $40 each
- Unlimited users on all plans
- API access requires Report or PRO subscription
- Claims-specific pricing based on monthly volume

**Verdict:** Strong professional tool with good API, but oriented toward field-service workflows (construction, insurance, remodeling) rather than photo-only reconstruction. The REST API is better documented than CubiCasa's but the core technology still requires on-site scanning.

---

### 2.3 Apple RoomPlan API

**What it does:** Swift framework that uses iPhone/iPad LiDAR scanner + camera to create parametric 3D room models. Detects walls, doors, windows, openings, and 16 categories of furniture. Outputs structured room data as parametric geometry.

**API availability:** YES -- native iOS/iPadOS Swift API
- Part of ARKit/RealityKit framework
- Two integration modes:
  - `RoomCaptureView`: Drop-in UI component with scanning guidance
  - `RoomCaptureSession`: Custom data API for programmatic access to live scan data
- Requires iOS 16+ and LiDAR-equipped device

**Input requirements:**
- Real-time LiDAR + camera scanning (MUST be done on-device)
- Requires LiDAR-equipped iPhone 12 Pro or later / iPad Pro
- Room size limit: 30 ft x 30 ft (9m x 9m)
- Minimum 50 lux lighting
- Scanning session should not exceed 5 minutes
- NOT compatible with static photos

**Output format:**
- USD/USDA/USDZ (Universal Scene Description) -- parametric 3D model
- Includes dimensions for each component (walls, cabinets, etc.)
- Furniture type classification
- Compatible with AutoCAD, Shapr3D, Cinema 4D for post-processing
- No direct DXF/SVG output

**Measurement accuracy:**
- Dimensional accuracy: "error usually staying below 5%"
- Object detection: 91% precision, 90% recall at 30% 3D IoU across 16 categories
- Best categories (fridge, bed, sofa): >93% precision and recall
- Hardest category (chairs): 83% precision, 87% recall

**Integration complexity for Python pipeline:**
- HIGH. RoomPlan is a Swift-only API running on iOS devices.
- Cannot be called from a Python backend directly.
- Would require building an iOS companion app that captures data and sends it to the Python pipeline.
- The USD output can be parsed in Python using OpenUSD libraries, but this adds significant complexity.
- Recent developer reports of degraded scanning quality (jittery overlays, poor results even for simple rooms).

**Non-convex room handling:** LIMITED
- RoomPlan reduces all geometry to rectangular primitives
- L-shaped sofas are handled by clipping two boxes at 90 degrees
- Non-convex room shapes are decomposed into rectangular segments, not captured as single coherent polygons
- Angled walls, arched openings, and curved surfaces are simplified to rectangles
- Developers report needing custom post-processing to preserve actual geometry

**Commercial licensing and pricing:**
- FREE -- included in iOS SDK, no per-use fees
- Requires Apple Developer Program membership ($99/year)
- No cloud processing costs
- All computation runs on-device

**Verdict:** Excellent free option with decent accuracy, but severely limited by Apple ecosystem lock-in (Swift + LiDAR device only). The rectangular primitive constraint means it cannot faithfully represent non-convex rooms. Integration with a Python backend is awkward. Best used as a data capture front-end with heavy post-processing.

---

### 2.4 Matterport API

**What it does:** Creates digital twins (full 3D models) of spaces from which schematic floor plans can be extracted. Primarily designed for 3D virtual tours with floor plan as a secondary output. Uses proprietary 3D cameras or compatible 360-degree cameras.

**API availability:** YES -- GraphQL API + JavaScript SDK
- GraphQL API at `https://api.matterport.com/`
- Showcase SDK (JavaScript, `@matterport/sdk` on npm) for embedding 3D tours
- Up to 5 API keys per account
- Admin-level API keys grant full account access
- Postman collection available
- Enterprise-only: Account API, Folders API

**Input requirements:**
- 3D scan using Matterport Pro camera ($3,395+), or compatible 360-degree camera, or iPhone/iPad LiDAR
- Requires capturing the space with specialized scanning workflow
- NOT static photos

**Output format:**
- PNG, SVG (native floor plan exports)
- PDF (combined multi-floor document)
- NO native DXF support -- requires third-party services (MP2FP, CAPTUR3D)
- OBJ mesh files, point cloud data via API
- Schematic floor plans are an add-on purchase ($15-$30 per plan)

**Measurement accuracy:**
- Linear wall-to-wall measurements: ~2% accuracy (not guaranteed for all environments)
- Described as "marketing schematic" -- not based on ANSI measurement standards
- Not suitable for appraisal-grade measurements without additional verification

**Integration complexity for Python pipeline:**
- MEDIUM-HIGH. The API is well-documented (GraphQL) but requires Matterport account and camera hardware.
- Python can make GraphQL requests easily, but floor plan extraction requires purchasing add-ons through the platform.
- The JavaScript SDK is client-side only (browser-based), not suitable for server-side Python processing.
- Floor plan data is not directly accessible via API -- you get 3D model data and must order floor plan rendering separately.

**Non-convex room handling:** YES -- 3D scanning captures arbitrary room geometry. The floor plan extraction handles complex shapes since it is derived from the full 3D model.

**Commercial licensing and pricing:**
- Subscription plans starting at $9.99/month
- Schematic floor plans: $15-$30 add-on per plan
- API/SDK access requires paid subscription
- Enterprise tier for advanced API features (Account API, Folders API)
- Matterport Pro 2 camera: ~$3,395
- Third-party DXF conversion services add additional cost

**Verdict:** Industry-leading 3D capture platform, but floor plans are a secondary feature. High total cost (camera + subscription + per-plan fees + third-party DXF conversion). Not suitable for a "photos-only" workflow. The GraphQL API is powerful for managing spaces but does not provide direct floor plan generation from uploaded images.

---

### 2.5 Planner 5D

**What it does:** AI-powered 2D/3D home design platform. Can recognize uploaded floor plan images/sketches and convert them to editable digital floor plans. Primarily a design tool, not a measurement tool.

**API availability:** NO public API
- Enterprise/B2B solutions available through their sales team
- White-label options with AR/VR integration
- No developer documentation publicly available

**Input requirements:**
- Manual drawing (drag-and-drop interface)
- Upload of existing floor plan image or hand-drawn sketch for AI recognition
- NOT from room photos -- only from existing floor plan images

**Output format:**
- 2D floor plan views
- 3D rendered views (up to 4K)
- 360-degree walkthroughs
- Limited export formats compared to CAD tools

**Measurement accuracy:** Not applicable -- this is a design tool, not a measurement tool. Users input dimensions manually.

**Integration complexity for Python pipeline:** NOT FEASIBLE without enterprise agreement. No public API.

**Non-convex room handling:** YES for design -- users can draw any room shape. NOT relevant for automated detection.

**Commercial licensing and pricing:**
- Free version with limited features
- Premium subscription for full catalog and rendering
- Enterprise: contact sales for custom pricing

**Verdict:** Not relevant to this project. Planner 5D is a design tool for creating floor plans from scratch or digitizing existing blueprints, not for reconstructing floor plans from room photos. No public API.

---

### 2.6 Floorfy and iGUIDE

#### Floorfy

**What it does:** Creates 3D virtual tours and automatic 2D floor plans from 360-degree camera captures. Generates commercial videos, HD photos, and property descriptions automatically.

**API availability:** YES (confirmed, limited details)
- Integrates with Facebook, Instagram, LinkedIn, Matterport, WordPress, YouTube
- Details of the floor plan generation API are not publicly documented

**Input requirements:** 360-degree camera photos of the property

**Output format:** 2D floor plans, 3D virtual tours, HD photos, commercial videos

**Measurement accuracy:** Not published

**Pricing:** Starting at $24/month, free trial available

**Verdict:** Primarily a real estate marketing tool. API exists but is poorly documented for developer integration. Not suitable for precision floor plan generation.

#### iGUIDE

**What it does:** Creates precise floor plans and 3D virtual tours using proprietary hardware (iGUIDE camera system). Focused on measurement accuracy and compliance.

**API availability:** LIMITED -- data export capabilities, not a full developer API
- Export formats: SVG, PDF, DWG, RVT, ESX, FML
- Integration with Floorplanner for 3D rendering
- Google Street View export

**Input requirements:** Capture using iGUIDE's proprietary camera system

**Output format:**
- SVG, PDF, DWG, RVT (Revit), ESX (Xactimate), FML
- Spatial reports (PDF & CSV) with room and object measurements

**Measurement accuracy:**
- RMS and ANSI compliant
- Distance measurement uncertainty: 0.5% or better
- Square footage uncertainty: 1% or better
- THE MOST ACCURATE option in this entire evaluation

**Non-convex room handling:** YES -- full room geometry from laser scanning

**Pricing:** No monthly fees; pay per processing. Own your data.

**Verdict:** Best measurement accuracy in the industry (0.5% error) but requires proprietary hardware. No public developer API for submitting images. Best for professional use cases where measurement precision is paramount. The DWG/SVG exports are directly usable in CAD workflows.

---

### 2.7 LLM Vision Models (GPT-4V / Claude Vision)

**What they do:** Multimodal large language models that can analyze room photos and describe layouts, estimate room configurations, and reason about spatial relationships. They do NOT generate floor plan drawings directly.

**API availability:** YES -- standard LLM APIs
- OpenAI API (GPT-4o, GPT-4.1) with vision input
- Anthropic API (Claude 4 Opus, Claude 4 Sonnet) with vision input
- Both accept base64-encoded images or URLs

**Input requirements:**
- Standard room photos (exactly what the current project uses)
- No special hardware, format, or scanning workflow required

**Output format:**
- Text descriptions of room layout
- Estimated dimensions (qualitative, not precise)
- Room type classification
- Spatial relationship descriptions
- Can generate code/SVG to draw floor plans (with significant prompt engineering)

**Measurement accuracy:**
- NOT suitable for dimensional measurement
- Can provide rough estimates ("this looks like a 12x15 foot room")
- Research shows GPT-4o and Claude 3.5 Sonnet can parse existing floor plan images for navigation
- Performance degrades with larger/more complex spaces
- No metric ground truth -- estimates are qualitative

**Integration complexity for Python pipeline:**
- LOW -- simple API calls with image payloads
- Could be used as an augmentation layer: "Given these 4 photos, describe the room layout, estimate room shape (rectangular, L-shaped, U-shaped), and identify doors/windows"
- Output requires parsing (structured JSON prompting recommended)

**Non-convex room handling:**
- Can identify L-shaped and U-shaped rooms from visual cues
- Cannot produce precise polygon coordinates
- Best used for room shape classification, not boundary extraction

**Commercial licensing and pricing:**
- OpenAI: ~$2.50-$10 per 1M input tokens (vision), $10-$30 per 1M output tokens
- Anthropic: similar pricing tiers
- Cost per room analysis: approximately $0.05-$0.50 depending on image count and model

**Verdict:** LLM vision models are NOT a replacement for geometric reconstruction, but they are a highly valuable augmentation. Best use case: feed room photos to GPT-4o or Claude 4 to classify room shape (rectangular vs. L-shaped), identify doors/windows/features, and estimate rough proportions. This information can then guide the geometric pipeline (e.g., "expect an L-shaped room with a door on the south wall"). The cost is negligible.

---

### 2.8 Academic Models (FloorNet, Floor-SP, PolyRoom, BADGR, FloorSAM)

#### FloorNet (ECCV 2018)

**What it does:** Turns RGBD videos of indoor spaces into vector-graphics floor plans using three neural network branches (PointNet for 3D points, CNN for top-down density, CNN for RGB images).

**Input:** RGBD video sequences
**Output:** Vector floor plan with walls and rooms
**Code:** https://github.com/art-programmer/FloorNet
**Dependencies:** Python 2.7, TensorFlow >=1.3, Gurobi (IP solver, academic-only license)
**Dataset:** 155 residential unit scans (135 train, 20 test)

**Limitations:** Old codebase (Python 2.7, TF 1.x), requires RGBD input (not standard RGB photos), Gurobi license needed. Historically important but superseded by newer methods.

#### Floor-SP (ICCV 2019)

**What it does:** Reconstructs floor plans from RGBD scans using sequential room-wise shortest path optimization. Uses Mask-RCNN for room instance segmentation and DRN for corner/edge likelihood estimation.

**Input:** RGBD scans
**Output:** Vector floor plan with room polygons
**Code:** https://github.com/woodfrog/floor-sp
**Approach:** Room-wise coordinate descent solving shortest path problems; objective function combines data terms (from DNNs), consistency terms (shared walls), and model complexity.

**Non-convex rooms:** YES -- the room-wise approach can reconstruct arbitrary room polygons.

**Limitations:** Requires RGBD input, older codebase, complex optimization pipeline.

#### PolyRoom (ECCV 2024) -- RECOMMENDED

**What it does:** Room-aware Transformer for floor plan reconstruction from point clouds. Uses uniform sampling representation, room-aware query initialization, and room-aware self-attention.

**Input:** Point clouds (which the current project already generates)
**Output:** Vectorized floor plan with room polygons
**Code:** https://github.com/3dv-casia/PolyRoom
**Key advantage:** Surpasses prior state-of-the-art both quantitatively and qualitatively. Modern architecture (Transformer-based). Directly accepts point cloud input.

**Non-convex rooms:** YES -- produces per-room polygons without convexity constraints.

**Limitations:** Research code (may need adaptation for production use), requires training data.

#### BADGR (CVPR 2025 Highlight) -- RECOMMENDED

**What it does:** A diffusion model that jointly performs floor plan reconstruction and bundle adjustment from wide-baseline RGB panoramas. Uses 1D floor boundary predictions from dozens of sparsely captured images.

**Input:** Wide-baseline RGB panoramas (closest to "photos" input)
**Output:** Camera poses and floor plan layouts (wall positions)
**Project page:** https://badgr-diffusion.github.io/
**Key advantages:**
- Accepts RGB images (not RGBD)
- Handles sparse, wide-baseline views
- Jointly optimizes poses and layout
- Learns structural constraints (wall adjacency, collinearity)
- Can inpaint occluded layouts
- CVPR 2025 Highlight paper -- cutting-edge quality

**Non-convex rooms:** YES -- the diffusion model reconstructs wall segments without convexity assumptions.

**Limitations:** Very recent (March 2025), code availability unclear, requires panoramic images (not standard perspective photos), from University of Washington research group.

#### FloorSAM (September 2025)

**What it does:** Integrates point cloud density maps with the Segment Anything Model (SAM) for floor plan reconstruction from LiDAR data. Uses grid-based filtering, adaptive resolution projection, and SAM's zero-shot segmentation.

**Input:** LiDAR point clouds
**Output:** Room segmentation masks and regularized contours
**Key advantage:** Leverages foundation model (SAM) for zero-shot room segmentation. Better accuracy, recall, and robustness than FloorSP, PolyRoom, and others on benchmark datasets.

**Non-convex rooms:** YES -- SAM-based segmentation handles arbitrary shapes.

**Limitations:** Requires LiDAR input, very recent paper, code availability uncertain.

---

### 2.9 Polycam

**What it does:** AI-powered 3D scanning app that creates detailed 3D models and floor plans from mobile devices. Uses LiDAR (Room Mode) or photos (Photo Mode) for different capture types.

**API availability:** NO public developer API
- Mobile app + web platform
- Enterprise solutions available via sales team
- GitHub organization has tools for raw data export (NeRF formats)
- Floor plan mode requires LiDAR

**Input requirements:**
- LiDAR scan for floor plans (LiDAR-equipped devices only)
- Photo mode for 3D object scanning (not floor plans)

**Output format:**
- OBJ, GLTF, FBX, STL, USDZ (3D meshes)
- DXF, PLY, LAS, XYZ (point cloud/CAD)
- 2D floor plans with interior measurements
- Spatial reports (PDF & CSV) with room/object measurements

**Measurement accuracy:** Not published, but leverages Apple's RoomPlan under the hood for Room Mode.

**Non-convex room handling:** Same as RoomPlan (rectangular primitives).

**Pricing:**
- Free tier with limited exports
- Pro: subscription for advanced features
- Business/Enterprise: advanced floor plans, measurements, team features

**Verdict:** Good mobile scanning tool, but no developer API and floor plans require LiDAR. Not suitable for integration into a Python pipeline. Essentially a consumer-friendly wrapper around Apple RoomPlan + photogrammetry.

---

### 2.10 RasterScan (Floor Plan Digitalization API)

**What it does:** Converts raster floor plan images (blueprints, hand-sketches) into structured vector formats and 3D models using deep learning and image processing.

**API availability:** YES -- REST API
- Raster to Vector (Base64): `POST https://backend.rasterscan.com/raster-to-vector-base64`
- Raster to Vector (Raw): `POST https://backend.rasterscan.com/raster-to-vector-raw`
- API key authentication via `x-api-key` header
- Available on RapidAPI
- On-premise Docker deployment: `docker pull rasterscan/floor-plan-recognition:latest-cpu`
- Free demo on Hugging Face

**Input requirements:**
- Floor plan IMAGES (blueprints, sketches, existing floor plans)
- Formats: JPEG, PNG, BMP
- NOT room photos -- requires an existing floor plan image as input

**Output format:**
- DXF, IFC, GLTF (vector/3D)
- SVG (vector)
- Structured JSON with walls, doors, symbols, room dimensions

**Measurement accuracy:** "High accuracy" claimed but no specific percentage published.

**Non-convex room handling:** YES -- recognizes arbitrary room shapes from blueprints.

**Pricing:** Custom/enterprise pricing, not publicly listed. Contact sales for quotes.

**Verdict:** Useful as a POST-PROCESSING tool if the current pipeline can generate a reasonable raster floor plan image. Could digitize the matplotlib output into vector format. However, it cannot generate a floor plan from room photos -- it only converts existing floor plan images. The on-premise Docker deployment is attractive for data privacy.

---

### 2.11 Floorplanner API

**What it does:** API for creating, editing, and rendering floor plans. Includes an ML service that converts floor plan images into editable Floorplanner projects.

**API availability:** YES -- REST API v2
- Sandbox and production environments (separate API keys)
- HTTP Basic Authentication
- ML service: converts floor plan images to Floorplanner projects (walls, doors, windows detected in near-realtime)
- Matterport conversion service
- Mobile SDK (Wescan) for point cloud capture

**Input requirements:**
- Manual creation via API endpoints
- OR floor plan image upload for ML conversion
- OR Matterport project import
- OR mobile point cloud (Wescan SDK)
- NOT directly from room photos

**Output format:** 2D & 3D floor plan images, editable projects

**Measurement accuracy:** Not published (design tool, not measurement tool).

**Pricing:** Not publicly documented. Contact for enterprise/API pricing.

**Verdict:** The ML-based floor plan image conversion service is interesting but limited to converting existing floor plan images, not generating from room photos. The Matterport conversion path could be useful in a multi-tool pipeline. API is well-documented for CRUD operations on floor plan projects.

---

### 2.12 MeasureSquare SDK

**What it does:** Web SDK for creating interactive, customizable floor plan models. Primarily designed for flooring industry (takeoffs, estimates, material calculations).

**API availability:** YES
- M2Diagram SDK (JavaScript/web) for embedding interactive floor plans
- M2Cloud API for syncing estimates, orders, diagrams
- Documentation on GitBook
- Publishable API Key required

**Input requirements:** Floor plan data created in M2 app (M2.NET or M2.Mobile), synchronized to M2Cloud.

**Output format:** Interactive web floor plans, integration with QuickBooks Online, Microsoft Excel, Salesforce.

**Measurement accuracy:** Professional-grade for flooring takeoffs (exact measurement not published).

**Pricing:** Starting at $149/month subscription.

**Verdict:** Specialized for the flooring industry. Not relevant for floor plan generation from photos. The SDK is for displaying/editing existing floor plans, not creating them from images.

---

### 2.13 Zillow Indoor Dataset + AI Floor Plans

**What it does:** Zillow uses AI to automatically generate interactive floor plans from panoramic photos captured by professional photographers. The Zillow Indoor Dataset (ZInD) is an open-source research dataset.

**API availability:**
- Zillow's AI floor plan generation: NO public API (internal to Zillow platform)
- ZInD dataset: YES, freely available for academic/non-commercial use
  - 71,474 panoramas from 1,524 unfurnished homes
  - Annotations: 3D room layouts, 2D/3D floor plans, panorama locations, windows/doors
  - ~40GB download from Bridge Platform
  - Apache License (code), custom terms (data)

**Input requirements (Zillow internal):** Panoramic photos from Zillow Media Experts professional photographers

**Output format (ZInD):** JSON annotations with room layouts, camera poses, floor plans

**Measurement accuracy:** Not published for Zillow's production system. ZInD ground truth took 1,500+ hours of annotation.

**Non-convex room handling:** ZInD includes non-Manhattan layouts (cuboid, Manhattan, and non-Manhattan distributions).

**Verdict:** ZInD is an excellent research resource for training and benchmarking floor plan reconstruction models. Zillow's production AI is not available as an API. The dataset includes ground truth for non-convex rooms, making it ideal for training/validating new algorithms.

---

## 3. Comparison Matrix

### 3.1 Feature Comparison

| Service | Accepts Photos | REST API | Python-Friendly | Non-Convex Rooms | Vector Output | Measurement Accuracy |
|---------|---------------|----------|-----------------|------------------|---------------|---------------------|
| **CubiCasa** | Video scan | YES | YES | YES | SVG, DXF | 95-97% (3-5% error) |
| **magicplan** | AR scan | YES | YES (libraries) | YES | PDF, Image | ~95%+ with laser |
| **Apple RoomPlan** | LiDAR scan | Swift only | NO | LIMITED (rectangles) | USD/USDZ | ~95% (5% error) |
| **Matterport** | 3D camera | GraphQL | YES | YES | SVG, PNG (no DXF) | ~98% (2% error) |
| **Planner 5D** | Blueprint image | NO | NO | YES (manual) | Limited | N/A (design tool) |
| **iGUIDE** | Laser scan | Limited | NO | YES | DWG, SVG, PDF | 99.5% (0.5% error) |
| **Floorfy** | 360 camera | YES (limited) | Unclear | Unclear | 2D plans | Not published |
| **GPT-4V/Claude** | ANY photos | YES | YES | YES (classification) | Text/code only | NOT suitable |
| **PolyRoom** | Point clouds | Open-source | YES (Python) | YES | Vector polygons | Research-grade |
| **BADGR** | RGB panoramas | Open-source | YES (Python) | YES | Wall positions | Research-grade |
| **FloorSAM** | LiDAR clouds | Open-source | YES (Python) | YES | Contours | State-of-the-art |
| **RasterScan** | Floor plan images | YES | YES | YES | DXF, SVG, IFC | High (not quantified) |
| **Floorplanner** | Floor plan images | YES | Partial | YES | 2D/3D images | N/A (design tool) |
| **Polycam** | LiDAR scan | NO | NO | LIMITED | DXF, OBJ | Not published |
| **MeasureSquare** | App data | YES (JS SDK) | NO | N/A | Web interactive | Professional |
| **ZInD (Zillow)** | Panoramas (dataset) | Dataset only | YES | YES | JSON annotations | Ground truth |

### 3.2 Weighted Scoring (Scale 1-10)

Criteria weights for this project:
- Accepts standard photos as input: 25%
- API/integration ease with Python: 20%
- Measurement accuracy: 20%
- Non-convex room support: 15%
- Vector output quality: 10%
- Cost effectiveness: 10%

| Service | Photos (25%) | Python API (20%) | Accuracy (20%) | Non-Convex (15%) | Vector (10%) | Cost (10%) | **Weighted Score** |
|---------|-------------|-----------------|----------------|------------------|-------------|-----------|-------------------|
| **CubiCasa** | 4 (video) | 8 | 9 | 9 | 8 | 7 | **7.15** |
| **BADGR** | 7 (panoramas) | 7 | 7 | 9 | 7 | 10 | **7.60** |
| **PolyRoom** | 6 (point clouds) | 8 | 7 | 9 | 8 | 10 | **7.55** |
| **magicplan** | 3 (AR scan) | 7 | 8 | 9 | 6 | 5 | **6.30** |
| **GPT-4V/Claude** | 10 | 10 | 2 | 7 | 2 | 9 | **6.65** |
| **RasterScan** | 2 (blueprints) | 9 | 7 | 8 | 9 | 7 | **6.30** |
| **Apple RoomPlan** | 2 (LiDAR) | 2 | 8 | 4 | 5 | 9 | **4.55** |
| **Matterport** | 2 (3D camera) | 6 | 9 | 8 | 6 | 3 | **5.45** |
| **FloorSAM** | 3 (LiDAR) | 6 | 8 | 9 | 7 | 10 | **6.45** |
| **Floorplanner** | 2 (blueprints) | 6 | 4 | 7 | 6 | 5 | **4.50** |
| **iGUIDE** | 1 (laser) | 2 | 10 | 9 | 9 | 4 | **5.10** |
| **Polycam** | 2 (LiDAR) | 1 | 6 | 4 | 7 | 6 | **3.75** |
| **Planner 5D** | 1 | 1 | 1 | 5 | 3 | 6 | **2.30** |

---

## 4. Ranking and Recommendations

### Final Ranking

| Rank | Option | Score | Best For |
|------|--------|-------|----------|
| 1 | **BADGR (academic)** | 7.60 | Closest to "photos in, floor plan out" with state-of-the-art quality |
| 2 | **PolyRoom (academic)** | 7.55 | Best for point-cloud-to-floorplan; directly compatible with current pipeline output |
| 3 | **CubiCasa (commercial)** | 7.15 | Best commercial option if input is changed from photos to video |
| 4 | **LLM Vision augmentation** | 6.65 | Best as augmentation layer for room shape classification and feature detection |
| 5 | **FloorSAM (academic)** | 6.45 | Best if LiDAR data is available |
| 6 | **magicplan (commercial)** | 6.30 | Best for field-service workflows with good API |
| 6 | **RasterScan (commercial)** | 6.30 | Best for digitizing existing floor plan outputs |

### Recommended Strategy

**Primary recommendation: Hybrid approach combining options 2, 4, and 6**

1. **PolyRoom** for the core reconstruction (point cloud to vectorized floor plan)
2. **LLM Vision (Claude/GPT)** for room shape classification and feature identification
3. **RasterScan** for post-processing/digitization if needed
4. **Shapely** for non-convex polygon operations

This approach:
- Preserves the current "photos as input" workflow
- Replaces the convex hull + Hough line pipeline with a state-of-the-art Transformer model
- Adds room shape intelligence via LLM vision at minimal cost
- Produces vector output compatible with CAD workflows
- Uses only open-source/low-cost components

---

## 5. Implementation Roadmaps

### 5.1 Option A: PolyRoom + LLM Vision + RasterScan (Recommended)

**Timeline:** 6-8 weeks
**Cost:** Minimal (open-source + LLM API costs ~$0.10-$0.50/room)

#### Step 1: Integrate PolyRoom (Weeks 1-3)

**What:** Replace `floor_plan_generator.py` convex hull + Hough pipeline with PolyRoom's Transformer-based reconstruction.

**Requirements:**
- Clone PolyRoom repository (https://github.com/3dv-casia/PolyRoom)
- Install dependencies (PyTorch, etc.)
- The current pipeline already generates point clouds -- these are PolyRoom's input format
- Adapt PolyRoom's inference code to accept point clouds from `room_reconstructor.py`

**Key integration points:**
- `room_reconstructor.py` produces point clouds via TSDF fusion or ICP alignment
- Feed these point clouds to PolyRoom's inference module
- PolyRoom outputs room polygon vertices in the ground plane
- Replace `FloorPlanGenerator._detect_room_boundary()` with PolyRoom output

**Milestones:**
1. Week 1: PolyRoom running standalone on sample point clouds
2. Week 2: Integration with current pipeline's point cloud output
3. Week 3: Validation against known room dimensions; tuning

**Potential obstacles:**
- PolyRoom may need fine-tuning on the project's point cloud quality (monocular depth vs. LiDAR)
- Research code may need adaptation for production use (error handling, edge cases)
- Point cloud density from monocular depth may be lower than PolyRoom expects

**Mitigation:**
- Start with PolyRoom's pre-trained weights
- If point cloud quality is insufficient, augment with Depth Pro (metric depth) to improve 3D accuracy
- Add point cloud densification step before feeding to PolyRoom

#### Step 2: Add LLM Vision Augmentation (Week 4)

**What:** Use GPT-4o or Claude 4 Vision to analyze input photos and provide room layout hints.

**Implementation:**
```
For each input image set:
  1. Send photos to LLM Vision API with structured prompt:
     "Analyze these room photos. For each photo, identify:
      - Room shape (rectangular, L-shaped, U-shaped, irregular)
      - Visible doors (count, approximate wall position)
      - Visible windows (count, approximate wall position)
      - Room type (bedroom, kitchen, bathroom, living room)
      - Approximate room proportions (wider than deep, square, etc.)
      Return JSON."
  2. Parse response
  3. Use as priors/constraints for PolyRoom reconstruction:
     - If LLM says "L-shaped", constrain polygon search accordingly
     - Door/window positions guide opening detection
     - Room type informs expected dimensions
```

**Cost:** ~$0.10-$0.50 per room analysis (4-5 photos)

**Milestones:**
1. Prompt engineering and validation against known room layouts
2. Integration with pipeline as pre-processing step
3. Evaluation of whether LLM hints improve reconstruction quality

#### Step 3: Vector Output with RasterScan or ezdxf (Weeks 5-6)

**What:** Produce DXF/SVG vector floor plans from PolyRoom's polygon output.

**Option A -- Direct rendering with ezdxf/svgwrite:**
- Convert PolyRoom room polygons to DXF/SVG using ezdxf and svgwrite
- Add wall thickness, dimension lines, scale bar, room labels
- This is the approach described in `FLOOR_PLAN_REVIEW.md` Phase 3

**Option B -- RasterScan post-processing:**
- Render PolyRoom output as a clean raster floor plan image
- Submit to RasterScan API for vectorization
- Get back DXF/SVG with walls, doors, symbols detected
- Advantage: RasterScan's ML adds door/window detection on the raster image

**Recommendation:** Use Option A (direct ezdxf/svgwrite) for full control, with Option B as validation.

**Milestones:**
1. Week 5: ezdxf renderer producing DXF files with walls and dimensions
2. Week 6: SVG renderer, scale bar, room labels, door/window symbols

#### Step 4: Validation and Refinement (Weeks 7-8)

- Test against rooms with known dimensions
- Compare accuracy against current pipeline (target: reduce from 20-30% error to <10%)
- Validate non-convex room handling (L-shaped, U-shaped test cases)
- Performance optimization (inference time target: <30 seconds per room)

**Success metrics:**
- Measurement accuracy: <10% error (vs. current 20-30%)
- Non-convex room support: L-shaped and U-shaped rooms correctly reconstructed
- Vector output: valid DXF files that open in AutoCAD/FreeCAD
- Processing time: <60 seconds for 5-image input

**Resource requirements:**
- GPU for PolyRoom inference (same as current Depth-Anything requirement)
- LLM API access (OpenAI or Anthropic account)
- Python packages: ezdxf, svgwrite, shapely, torch

---

### 5.2 Option B: CubiCasa Integration (Commercial Path)

**Timeline:** 3-4 weeks
**Cost:** $22.99-$29.99 per floor plan + development time

#### Step 1: Set Up CubiCasa Developer Account (Week 1)

**What:** Register for CubiCasa developer access, get API keys, set up staging environment.

**Actions:**
1. Create account at https://qa-customers.cubi.casa (staging)
2. Generate API key from company developer page
3. Review API documentation at https://integrate.docs.cubi.casa/
4. Review Conversion API at https://cubicasaconversionapi.docs.apiary.io/
5. Install CubiCasa iOS SDK via SPM or Android SDK AAR

**Milestones:**
1. Staging API key obtained
2. Successful test API call (list projects, create project)

#### Step 2: Build Mobile Capture Integration (Weeks 2-3)

**What:** Either build a lightweight mobile app using CubiCasa SDK, or modify the workflow to accept video captured by CubiCasa's app.

**Option A -- Use CubiCasa's own app:**
- Users scan with CubiCasa app
- Use Integrate API to pull completed floor plans into the Python pipeline
- Simplest approach, least development effort

**Option B -- Custom app with CubiCasa SDK:**
- Embed CubiCapture SDK in a custom iOS/Android app
- Upload scan zip to own server
- Forward to CubiCasa Conversion API
- More control but significantly more development effort

**Recommendation:** Start with Option A (use CubiCasa's app + Integrate API).

**Integration code (Python):**
```
1. User scans room with CubiCasa app
2. CubiCasa processes scan (6-24 hours)
3. Python pipeline polls Integrate API for completed projects
4. Downloads SVG/DXF floor plan
5. Parses floor plan into internal representation
6. Augments with 3D visualization from current pipeline
```

**Milestones:**
1. Week 2: End-to-end flow working on staging (scan -> API poll -> download)
2. Week 3: Production environment, error handling, Gradio UI integration

#### Step 3: Integration with Current Pipeline (Week 4)

**What:** Connect CubiCasa output to existing 3D visualization pipeline.

**Actions:**
- Parse CubiCasa SVG/DXF output for room geometry
- Use room polygon data to constrain/replace current point cloud boundary detection
- Display CubiCasa floor plan alongside 3D visualization in Gradio UI
- Add download buttons for DXF/SVG formats

**Milestones:**
1. CubiCasa floor plan displayed in Gradio alongside 3D view
2. Room measurements from CubiCasa shown in measurements panel
3. Download buttons for all output formats

**Success metrics:**
- Measurement accuracy: 3-5% (CubiCasa's stated accuracy)
- Non-convex room support: fully handled by CubiCasa
- Vector output: direct from CubiCasa (SVG, DXF)
- End-to-end time: depends on CubiCasa processing (6-24 hours, or 6 hours with add-on)

**Potential obstacles:**
- CubiCasa requires video walkthrough, not static photos -- this is a fundamental workflow change
- Processing time is 6-24 hours (not real-time)
- Per-plan cost adds up at scale
- API is self-described as "work in progress"

**Mitigation:**
- Frame the workflow change as "enhanced scanning mode" in the app
- Use the 6-hour expedited option for faster turnaround
- Negotiate volume pricing at 20+ scans/month
- Maintain current pipeline as a "quick preview" with CubiCasa as "precision mode"

---

## 6. Sources

### Commercial Services
- [CubiCasa Developer Portal](https://www.cubi.casa/developers/)
- [CubiCasa Integrate API Documentation](https://integrate.docs.cubi.casa/)
- [CubiCasa Accuracy FAQ](https://help.cubi.casa/en/articles/6662584-how-accurate-are-your-plans)
- [CubiCasa Pricing](https://www.cubi.casa/pricing/)
- [CubiCasa & Matterport Accuracy Testing](https://www.insiderealestatephotography.com/post/cubicasa-matterport-floor-plans-how-accurate-are-they)
- [magicplan REST API Documentation](https://apidocs.magicplan.app/)
- [magicplan API Reference](https://apidocs.magicplan.app/reference)
- [magicplan Pricing](https://magicplan.app/pricing)
- [magicplan Export Formats](https://help.magicplan.app/export-formats)
- [Matterport API Reference](https://api.matterport.com/)
- [Matterport SDK Documentation](https://matterport.github.io/showcase-sdk/api_home.html)
- [Matterport Measurement Accuracy](https://support.matterport.com/s/article/How-accurate-are-dimensions-in-Matterport-Spaces?language=en_US)
- [Matterport Floor Plan FAQ](https://support.matterport.com/s/article/FAQ-Schematic-Floor-Plans?language=en_US)
- [Planner 5D API Availability](https://support.planner5d.com/en/articles/7245729-is-there-a-public-api-available-for-planner-5d)
- [iGUIDE Floor Plans](https://goiguide.com/iguide/floor-plans)
- [Polycam Floor Plans](https://poly.cam/floor-plans)
- [RasterScan Platform](https://www.rasterscan.com/)
- [RasterScan GitHub (On-Premise Docker)](https://github.com/RasterScan/Floor-Plan-Recognition)
- [RasterScan Hugging Face Demo](https://huggingface.co/spaces/RasterScan/Automated-Floor-Plan-Digitalization)
- [Floorplanner API Documentation](https://floorplanner.readme.io/reference/api)
- [MeasureSquare SDK Documentation](https://diagram.measuresquare.com/document/index.html)

### Apple RoomPlan
- [RoomPlan Developer Documentation](https://developer.apple.com/documentation/roomplan/)
- [RoomPlan Overview](https://developer.apple.com/augmented-reality/roomplan/)
- [RoomPlan ML Research Paper](https://machinelearning.apple.com/research/roomplan)
- [RoomPlan Framework Review (it-jim)](https://www.it-jim.com/blog/roomplan-framework-by-apple/)
- [RoomPlan API Integration Guide](https://www.it-jim.com/blog/apple-roomplan-api/)
- [RoomPlan How-To Guide (iTechCraft)](https://itechcraft.com/blog/your-101-guide-to-using-apple-roomplan-api-for-your-next-app/)

### Academic Models
- [FloorNet (ECCV 2018)](https://art-programmer.github.io/floornet.html) | [GitHub](https://github.com/art-programmer/FloorNet)
- [Floor-SP (ICCV 2019)](https://arxiv.org/abs/1908.06702) | [GitHub](https://github.com/woodfrog/floor-sp)
- [PolyRoom (ECCV 2024)](https://arxiv.org/abs/2407.10439) | [GitHub](https://github.com/3dv-casia/PolyRoom)
- [BADGR (CVPR 2025 Highlight)](https://badgr-diffusion.github.io/) | [arXiv](https://arxiv.org/abs/2503.19340)
- [FloorSAM (September 2025)](https://arxiv.org/abs/2509.15750)
- [TF2DeepFloorplan (Flask API)](https://github.com/zcemycl/TF2DeepFloorplan)

### Datasets
- [Zillow Indoor Dataset (ZInD)](https://github.com/zillow/zind)
- [ZInD Paper (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Cruz_Zillow_Indoor_Dataset_Annotated_Floor_Plans_With_360deg_Panoramas_and_CVPR_2021_paper.pdf)
- [Zillow AI Floor Plans & Virtual Staging](https://zillow.mediaroom.com/2025-09-10-Zillow-brings-AI-powered-Virtual-Staging-to-Showcase-listings)
- [Redfin CubiCasa Integration](https://www.redfin.com/news/press-releases/redfin-integrates-cubicasa-floor-plans-to-enhance-home-search/)

### LLM Vision Research
- [VLMs for Floor Plan Parsing (arXiv, 2024)](https://arxiv.org/html/2409.12842)
- [Claude Sonnet for Architecture (ArchiLabs)](https://archilabs.ai/posts/anthropic-claude-sonnet-45-for-architectural-design)
- [GPT-4o Vision Benchmark (ICLR 2025)](https://openreview.net/forum?id=h3unlS2VWz)

### Python Libraries
- [Shapely (Non-Convex Polygons)](https://shapely.readthedocs.io/)
- [DeepFloorplan (DNN Floor Plan Recognition)](https://github.com/zlzeng/DeepFloorplan)
- [MLStructFP (Floor Plan Dataset)](https://pypi.org/project/MLStructFP/)
- [Non-Rectangular Room Generation (ScienceDirect, 2023)](https://www.sciencedirect.com/science/article/pii/S1524070323000061)
- [Best AI Floor Plan Generators Compared (CubiCasa, 2026)](https://www.cubi.casa/best-ai-floor-plan/)
