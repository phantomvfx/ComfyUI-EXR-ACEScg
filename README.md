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

### 2. Save EXR (`SaveEXRNode`)
Saves ComfyUI image tensors to 32-bit float EXR files.

**Parameters:**
- **images**: The ComfyUI image tensor to save.
- **filename_prefix**: The prefix for the saved EXR files.
- **color_space**:
  - `Raw (Keep Current)`: Saves the image as-is.
  - `Convert Linear sRGB to ACEScg`: Applies a matrix transformation converting ComfyUI's standard Linear sRGB outputs into ACEScg (AP1) color space before saving to EXR, ideal for VFX and compositing pipelines (like Nuke).

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/phantomvfx/ComfyUI-EXR-ACEScg.git
   ```
3. Ensure you have the required dependencies (primarily `opencv-python`, `torch`, and `numpy` which should already be included with your ComfyUI installation).

## Why ACEScg?

Standard ComfyUI uses sRGB/Linear workflows. When integrating AI-generated imagery into VFX pipelines (e.g., Nuke, Resolve, Maya), maintaining accurate scene-linear data using ACEScg is critical. This node allows you to easily ingest 32-bit EXRs rendered in ACEScg, process them in ComfyUI using standard linear sRGB, and then accurately convert them back to ACEScg EXRs to maintain your color pipeline perfectly intact.
