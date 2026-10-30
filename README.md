# ComfyUI-EXR-ACEScg

A custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides support for reading and writing EXR files with ACEScg color space management.

## Features

This extension adds two nodes under the `image/EXR` category:

### 1. Load EXR (`LoadEXRNode`)
Loads EXR images or image sequences into ComfyUI as standard image tensors.

**Parameters:**
- **image_path**: Path to your EXR file. Supports padding formats for sequences (e.g., `image.####.exr` or `image.%04d.exr`).
- **color_space**: 
  - `Raw (Keep ACEScg)`: Loads the EXR data without applying any color transformations.
  - `Convert ACEScg to Linear sRGB`: Applies a matrix transformation to convert ACEScg (AP1) color space to Linear sRGB, matching typical ComfyUI image workflows.
- **load_mode**: Choose between `single_frame` and `image_sequence`.
- **frame_index**: The starting frame index to load (especially useful for sequences or padded single files).
- **sequence_limit**: Limits the maximum number of frames to load when working with sequences (0 means load all available following `frame_index`).

**Outputs:**
- **image**: RGB, 32-bit float, scene-linear (no clipping). Channels are picked by name (`R,G,B`, else `Y`), so multi-layer EXRs load their main layer.
- **alpha**: the EXR's straight alpha (1 = opaque) as a `MASK`; all ones when the file has no alpha. Note this is the inverse of the mask that ComfyUI's built-in Load Image outputs.

The image is always returned at the **display window** size. Files with a smaller data window (Nuke bbox / autocrop) are padded with black and alpha 0; overscan outside the display window is cropped. The node re-runs automatically when a file changes on disk.

### 2. Save EXR (`SaveEXRNode`)
Saves ComfyUI image tensors to 32-bit float EXR files.

**Parameters:**
- **images**: The ComfyUI image tensor to save.
- **filename_prefix**: The prefix for the saved EXR files.
- **color_space**:
  - `Raw (Keep Current)`: Saves the image as-is.
  - `Convert Linear sRGB to ACEScg`: Applies a matrix transformation converting ComfyUI's standard Linear sRGB outputs into ACEScg (AP1) color space before saving to EXR, ideal for VFX and compositing pipelines (like Nuke).
- **alpha** (optional `MASK`): straight alpha; when connected the file is written as RGBA.
- **precision** (optional): `32-bit float` (default) or `16-bit half`.
- **compression** (optional): `zip` (default), `zips`, `piz`, `dwaa`, `dwab`, `pxr24`, `b44`, `rle` or `none`.

Files are named `<prefix>_00001.exr`, `<prefix>_00002.exr`, ... and never overwrite earlier saves, so the output reads directly as a `<prefix>_#####.exr` sequence in Nuke or Resolve.

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/phantomvfx/ComfyUI-EXR-ACEScg.git
   ```
3. Install the dependency into the ComfyUI Python environment:
   ```bash
   pip install -r ComfyUI-EXR-ACEScg/requirements.txt
   ```
   EXR I/O uses [OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO). Earlier versions used OpenCV, but many OpenCV wheels (for example `opencv-contrib-python` 5.x) are built without OpenEXR, and several OpenCV packages in one venv can shadow each other.

## Tests

Run from the repository root with the ComfyUI venv (stdlib `unittest`, no pytest needed). `COMFYUI_ROOT` points at your ComfyUI checkout:
```bash
COMFYUI_ROOT=/path/to/ComfyUI /path/to/ComfyUI/venv/bin/python -m unittest discover -s tests -v
```

## Why ACEScg?

Standard ComfyUI uses sRGB/Linear workflows. When integrating AI-generated imagery into VFX pipelines (e.g., Nuke, Resolve, Maya), maintaining accurate scene-linear data using ACEScg is critical. This node allows you to easily ingest 32-bit EXRs rendered in ACEScg, process them in ComfyUI using standard linear sRGB, and then accurately convert them back to ACEScg EXRs to maintain your color pipeline perfectly intact.
