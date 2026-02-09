# PDF Technology Usage Audit

Source analyzed: `3D-Room-Reconstruction-DeepResearch.pdf`  
Repository analyzed: `Room-Reconstruction-Demo`  
Date: 2026-02-09

Legend:
- `[x]` = used in this project
- `[ ]` = not used in this project
- `[~]` = partially used / adjacent implementation

## Technology and Library Matrix

| Technology / library named in PDF | Type | Used in this repo? | Evidence in repo | Reason |
|---|---|---|---|---|
| COLMAP (engine / SfM+MVS stack) | Open-source photogrammetry toolkit | [x] | `modules/sfm_processor.py`, `config.py` | Core reconstruction uses pycolmap bindings that call COLMAP SfM/MVS APIs. |
| pycolmap | Open-source Python binding | [x] | `requirements.txt`, `modules/sfm_processor.py` | Required dependency and directly imported/called for feature extraction, matching, mapping, undistort, patch-match, fusion. |
| Open3D | Open-source 3D processing library | [x] | `requirements.txt`, `modules/dense_reconstructor.py`, `modules/visualizer_3d.py`, `modules/room_reconstructor.py` | Used for TSDF fusion, point cloud/mesh operations, and visualization/export. |
| DPT (Depth Prediction Transformer) | Open-source depth model family | [x] | `config.py`, `modules/depth_estimator.py` | DPT is implemented as fallback depth model (`Intel/dpt-large`) through HuggingFace Transformers. |
| MiDaS | Open-source monocular depth family | [~] | `modules/depth_estimator.py` | Code references "DPT/MiDaS" conceptually; implementation uses HF depth models (Depth-Anything/DPT), not the MiDaS Git repo pipeline directly. |
| PyTorch | Open-source deep learning framework | [x] | `requirements.txt`, `modules/depth_estimator.py` | Required and used for depth-model inference. |
| Meshroom (AliceVision) | Open-source photogrammetry app | [ ] | (no import/usage) | Mentioned in PDF as alternative; no Meshroom CLI integration in codebase. |
| AliceVision (via Meshroom) | Open-source photogrammetry framework | [ ] | (no import/usage) | Not integrated; only discussed in PDF. |
| OpenMVG | Open-source SfM library | [ ] | (no import/usage) | Not present in dependencies or runtime modules. |
| OpenMVS | Open-source MVS/meshing toolkit | [ ] | (no import/usage) | Not present in dependencies or runtime modules. |
| Detectron2 | Open-source segmentation/detection | [ ] | (no import/usage) | No object/material segmentation pipeline implemented. |
| MIT SceneParse / scene parsing models | Research/model family | [ ] | (no import/usage) | Not present in dependencies or modules. |
| Segment Anything (SAM) | Open-source segmentation model | [ ] | (no import/usage) | Not integrated in current pipeline. |
| Mask R-CNN | Detection/segmentation model family | [ ] | (no import/usage) | Not integrated in current pipeline. |
| Instant-NGP (Instant NeRF) | Open-source neural rendering toolkit | [ ] | (no import/usage) | Mentioned as future/advanced option only. |
| NeRF pipeline (general) | Neural rendering approach | [ ] | (no import/usage) | No NeRF training/inference path in current code. |
| Gaussian Splatting | Neural rendering/reconstruction method | [ ] | (no import/usage) | Mentioned in PDF future directions; not implemented here. |
| MeshLab | External GUI tool | [ ] | (no import/usage) | Not programmatically integrated. |
| PyMeshLab | Python binding for MeshLab | [ ] | (no dependency/import) | Not used by this project. |
| Blender | External DCC tool | [ ] | (no import/usage) | Not integrated into pipeline. |
| CloudCompare | External point-cloud tool | [ ] | (no import/usage) | Not integrated into pipeline. |
| FloorNet | Floor-plan extraction research model | [ ] | (no import/usage) | Mentioned as idea; absent from implementation. |
| obj2gltf | Conversion tool | [ ] | (no import/usage) | No model conversion stage uses this tool. |
| Matterport | Commercial platform | [ ] | (no SDK/integration) | Mentioned as commercial alternative, not integrated. |
| Kaarta | Commercial hardware/software | [ ] | (no SDK/integration) | Mentioned as commercial alternative, not integrated. |
| Pix4D | Commercial software | [ ] | (no SDK/integration) | Mentioned as commercial alternative, not integrated. |
| Agisoft Metashape | Commercial software | [ ] | (no SDK/integration) | Mentioned as commercial alternative, not integrated. |
| Polycam | Commercial/mobile app | [ ] | (no SDK/integration) | Mentioned as commercial alternative, not integrated. |

## Notes

- The current implementation is centered on `pycolmap`/COLMAP for SfM and optional dense MVS.
- The current implementation is centered on `Open3D` for fusion, geometry processing, and visualization/export.
- The current implementation is centered on `PyTorch + Transformers` for monocular depth estimation (Depth-Anything/DPT path).
- Segmentation stack (Detectron2/SAM/Mask R-CNN), alternative photogrammetry stacks (Meshroom/OpenMVG/OpenMVS), and commercial platforms are not currently wired into runtime code.
