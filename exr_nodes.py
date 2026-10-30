import os
import re

import torch
import numpy as np
import folder_paths

# EXR I/O goes through OpenImageIO. OpenCV wheels are often built without OpenEXR
# (e.g. opencv-contrib-python 5.x), and several cv2 packages can shadow each other
# in one venv, so cv2 is not a dependable EXR reader/writer inside ComfyUI.
try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None

# ACEScg (AP1) to Linear sRGB Matrix
ACESCG_TO_SRGB_MATRIX = np.array([
    [ 1.7050796, -0.6218677, -0.0832119],
    [-0.1302559,  1.1408028, -0.0105469],
    [-0.0240075, -0.1289676,  1.1529751]
], dtype=np.float32)

# Linear sRGB to ACEScg (AP1) Matrix
SRGB_TO_ACESCG_MATRIX = np.linalg.inv(ACESCG_TO_SRGB_MATRIX)

EXR_COMPRESSIONS = ["zip", "zips", "piz", "dwaa", "dwab", "pxr24", "b44", "rle", "none"]

def require_oiio():
    if oiio is None:
        raise ImportError("ComfyUI-EXR-ACEScg needs OpenImageIO: pip install OpenImageIO (in the ComfyUI venv)")

def apply_color_matrix(image_np, matrix):
    flat_img = image_np.reshape(-1, 3)
    converted = np.dot(flat_img, matrix.T)
    return converted.reshape(image_np.shape)

def pick_channels(names):
    # Returns (rgb channel indices, alpha index or None), matched by name so that
    # multi-layer files (diffuse.R, depth.Z, ...) still pick the main rgba layer
    names = list(names)
    if all(c in names for c in ("R", "G", "B")):
        rgb = [names.index(c) for c in ("R", "G", "B")]
    elif "Y" in names:
        rgb = [names.index("Y")] * 3
    else:
        rgb = list(range(min(3, len(names))))
        rgb += [rgb[-1]] * (3 - len(rgb))
    alpha = names.index("A") if "A" in names else None
    return rgb, alpha

def read_exr(file_path):
    # Returns float32 RGB [H, W, 3] and alpha [H, W] covering the display window.
    # Pixels outside the data window (Nuke bbox / autocrop) are black with alpha 0;
    # overscan outside the display window is cropped.
    require_oiio()
    buf = oiio.ImageBuf(file_path)
    if buf.has_error:
        raise ValueError(f"OpenImageIO failed to load EXR image {file_path}: {buf.geterror()}")
    spec = buf.spec()
    rgb_idx, alpha_idx = pick_channels(spec.channelnames)

    full = buf.roi_full
    data = buf.roi
    if data != full:
        buf = oiio.ImageBufAlgo.crop(buf, full)
        if buf.has_error:
            raise ValueError(f"OpenImageIO failed to crop {file_path} to its display window: {buf.geterror()}")

    pixels = buf.get_pixels(oiio.FLOAT, full)
    if pixels is None:
        raise ValueError(f"OpenImageIO failed to read pixels from {file_path}: {oiio.geterror()}")
    pixels = pixels.reshape(full.height, full.width, -1)

    rgb = pixels[:, :, rgb_idx]
    if alpha_idx is not None:
        alpha = pixels[:, :, alpha_idx]
    else:
        # Opaque where the file has data, transparent in the padded area
        alpha = np.zeros((full.height, full.width), dtype=np.float32)
        x0, x1 = max(data.xbegin, full.xbegin) - full.xbegin, min(data.xend, full.xend) - full.xbegin
        y0, y1 = max(data.ybegin, full.ybegin) - full.ybegin, min(data.yend, full.yend) - full.ybegin
        if x1 > x0 and y1 > y0:
            alpha[y0:y1, x0:x1] = 1.0
    return np.ascontiguousarray(rgb, dtype=np.float32), np.ascontiguousarray(alpha, dtype=np.float32)

def write_exr(file_path, pixels, precision="float", compression="zip"):
    # pixels is float32 [H, W, 3] (RGB) or [H, W, 4] (RGBA)
    require_oiio()
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, precision)
    spec.channelnames = ("R", "G", "B", "A")[:channels]
    if channels == 4:
        spec.alpha_channel = 3
    spec.attribute("compression", compression)
    out = oiio.ImageOutput.create(file_path)
    if out is None:
        raise IOError(f"OpenImageIO cannot create an EXR writer for {file_path}: {oiio.geterror()}")
    try:
        if not out.open(file_path, spec) or not out.write_image(np.ascontiguousarray(pixels, dtype=np.float32)):
            raise IOError(f"OpenImageIO failed to write EXR file {file_path}: {out.geterror()}")
    finally:
        out.close()

class LoadEXRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "C:/path/to/your/image.####.exr", "multiline": False}),
                "color_space": (["Raw (Keep ACEScg)", "Convert ACEScg to Linear sRGB"],),
            },
            "optional": {
                "load_mode": (["single_frame", "image_sequence"], {"default": "single_frame"}),
                "frame_index": ("INT", {"default": 1, "min": 0, "max": 999999, "step": 1}),
                "sequence_limit": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            }
        }

    CATEGORY = "image/EXR"
    # alpha is straight EXR alpha (1 = opaque), all ones when the file has none.
    # Note this is the inverse of the MASK that ComfyUI's LoadImage outputs.
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "load_exr"

    @classmethod
    def IS_CHANGED(cls, image_path, color_space, load_mode="single_frame", frame_index=1, sequence_limit=0):
        # Re-run when any file on disk changes, not only when the inputs change
        try:
            files = cls()._collect_files(image_path, load_mode, frame_index, sequence_limit)
        except FileNotFoundError:
            return float("nan")
        return ";".join(f"{f}:{os.path.getmtime(f) if os.path.exists(f) else 'missing'}" for f in files)

    def _resolve_path(self, path_pattern, frame):
        # Nuke style ####
        def replace_hash(match):
            length = len(match.group(0))
            return f"{frame:0{length}d}"
        path = re.sub(r'#+', replace_hash, path_pattern)

        # Printf style %04d
        def replace_printf(match):
            format_str = match.group(0)
            return format_str % frame
        path = re.sub(r'%\d*d', replace_printf, path)

        return path

    def _resolve_input_path(self, target_path):
        if not os.path.isabs(target_path):
            input_dir = folder_paths.get_input_directory()
            potential_path = os.path.join(input_dir, target_path)
            if os.path.exists(potential_path):
                return potential_path
        return target_path

    def _get_sequence_files(self, path_pattern, start_frame=0, limit=0):
        # Resolve to directory
        folder, filename = os.path.split(path_pattern)
        if not folder:
            folder = folder_paths.get_input_directory()
        elif not os.path.isabs(folder):
            folder = os.path.join(folder_paths.get_input_directory(), folder)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Build regex for matching frames
        parts = re.split(r'(#+|%\d*d)', filename)
        regex_str = "^"
        for part in parts:
            if re.match(r'^#+$', part) or re.match(r'^%\d*d$', part):
                regex_str += r'(\d+)'
            else:
                regex_str += re.escape(part)
        regex_str += "$"
        regex = re.compile(regex_str)

        matches = []
        for f in os.listdir(folder):
            m = regex.match(f)
            if m:
                # Use the first captured group as frame number
                frame_num = int(m.group(1))
                matches.append((frame_num, os.path.join(folder, f)))

        matches.sort()
        matches = [m for m in matches if m[0] >= start_frame]

        if limit > 0:
            matches = matches[:limit]

        return [m[1] for m in matches]

    def _collect_files(self, image_path, load_mode, frame_index, sequence_limit):
        image_path = image_path.strip('"').strip("'")
        is_padded = bool(re.search(r'#+|%\d*d', image_path))

        if load_mode == "image_sequence" and is_padded:
            files = self._get_sequence_files(image_path, start_frame=frame_index, limit=sequence_limit)
            if not files:
                raise FileNotFoundError(f"No sequence found for pattern: {image_path} starting at frame {frame_index}")
            return files

        if load_mode == "single_frame" and is_padded:
            image_path = self._resolve_path(image_path, frame_index)
        return [self._resolve_input_path(image_path)]

    def load_exr(self, image_path, color_space, load_mode="single_frame", frame_index=1, sequence_limit=0):
        files_to_load = self._collect_files(image_path, load_mode, frame_index, sequence_limit)

        tensors = []
        alphas = []
        for file_path in files_to_load:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"EXR file not found at path: {file_path}")

            # Read EXR as 32-bit float RGB + separate alpha
            img, alpha = read_exr(file_path)

            # Apply Color Space Conversion if requested (RGB only, alpha is untouched)
            if color_space == "Convert ACEScg to Linear sRGB":
                img = apply_color_matrix(img, ACESCG_TO_SRGB_MATRIX)

            if tensors and img.shape != tuple(tensors[0].shape[1:]):
                raise ValueError(
                    f"Frame size mismatch in sequence: {file_path} is {img.shape[1]}x{img.shape[0]}, "
                    f"expected {tensors[0].shape[2]}x{tensors[0].shape[1]}"
                )

            # Convert to ComfyUI standard Tensor [1, H, W, C] and MASK [1, H, W]
            tensors.append(torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0))
            alphas.append(torch.from_numpy(alpha).unsqueeze(0))

        if not tensors:
            raise ValueError("No images loaded.")

        # Concatenate on batch dimension
        return (torch.cat(tensors, dim=0), torch.cat(alphas, dim=0))


class SaveEXRNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI_EXR"}),
                "color_space": (["Raw (Keep Current)", "Convert Linear sRGB to ACEScg"],),
            },
            # New widgets go last: saved UI workflows store widget values by position
            "optional": {
                # Straight alpha (1 = opaque). When connected, files are written as RGBA.
                "alpha": ("MASK", ),
                "precision": (["32-bit float", "16-bit half"], {"default": "32-bit float"}),
                "compression": (EXR_COMPRESSIONS, {"default": "zip"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "image/EXR"

    def save_exr(self, images, filename_prefix="ComfyUI_EXR", color_space="Raw (Keep Current)", alpha=None,
                 precision="32-bit float", compression="zip", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        if alpha is not None and alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        pixel_format = "half" if precision == "16-bit half" else "float"

        results = list()

        for batch_index, image in enumerate(images):
            img_np = image.cpu().numpy().astype(np.float32)

            if color_space == "Convert Linear sRGB to ACEScg":
                img_np = apply_color_matrix(img_np, SRGB_TO_ACESCG_MATRIX)

            if alpha is not None:
                alpha_np = alpha[min(batch_index, alpha.shape[0] - 1)].cpu().numpy().astype(np.float32)
                if alpha_np.shape != img_np.shape[:2]:
                    raise ValueError(
                        f"Alpha size {alpha_np.shape[1]}x{alpha_np.shape[0]} does not match "
                        f"image size {img_np.shape[1]}x{img_np.shape[0]}"
                    )
                img_np = np.dstack([img_np, alpha_np])

            # Plain zero padding: Nuke/Resolve read it as a name_#####.exr sequence, and
            # ComfyUI's counter scan parses it (the old "0_001" form made every run restart at 1)
            file_name = f"{filename}_{counter:05}.exr"
            file_path = os.path.join(full_output_folder, file_name)

            write_exr(file_path, img_np, pixel_format, compression)

            results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
