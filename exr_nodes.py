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

# ACES2065-1 (AP0) to ACEScg (AP1), from the ACES reference
AP0_TO_AP1_MATRIX = np.array([
    [ 1.4514393161, -0.2365107469, -0.2149285693],
    [-0.0765537734,  1.1762296998, -0.0996759264],
    [ 0.0083161484, -0.0060324498,  0.9977163014]
], dtype=np.float32)

# EXR `chromaticities` header values: red, green, blue and white xy
CHROMATICITIES = {
    "ACEScg": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "ACES2065-1": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
    "Rec.709": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
}
# Load EXR "Auto": how to reach linear sRGB from each known set of primaries.
# Files without the attribute are taken as ACEScg, the plates this pack is for.
AUTO_TO_SRGB = {
    None: ACESCG_TO_SRGB_MATRIX,
    "ACEScg": ACESCG_TO_SRGB_MATRIX,
    "ACES2065-1": ACESCG_TO_SRGB_MATRIX @ AP0_TO_AP1_MATRIX,
    "Rec.709": None,
}

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

def exr_primaries(file_path):
    # Name of the primaries in the file's `chromaticities` header (a CHROMATICITIES key),
    # None when the attribute is missing, or the raw values when they are not a known set
    require_oiio()
    inp = oiio.ImageInput.open(file_path)
    if inp is None:
        raise ValueError(f"OpenImageIO failed to open {file_path}: {oiio.geterror()}")
    try:
        chroma = inp.spec().getattribute("chromaticities")
    finally:
        inp.close()
    if chroma is None:
        return None
    chroma = tuple(float(v) for v in chroma)
    for name, ref in CHROMATICITIES.items():
        if max(abs(a - b) for a, b in zip(chroma[:6], ref[:6])) < 0.005:
            return name
    return chroma

def write_exr(file_path, pixels, precision="float", compression="zip", chromaticities=None):
    # pixels is float32 [H, W, 3] (RGB) or [H, W, 4] (RGBA)
    require_oiio()
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, precision)
    spec.channelnames = ("R", "G", "B", "A")[:channels]
    if channels == 4:
        spec.alpha_channel = 3
    spec.attribute("compression", compression)
    if chromaticities is not None:
        spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), CHROMATICITIES[chromaticities])
    out = oiio.ImageOutput.create(file_path)
    if out is None:
        raise IOError(f"OpenImageIO cannot create an EXR writer for {file_path}: {oiio.geterror()}")
    try:
        if not out.open(file_path, spec) or not out.write_image(np.ascontiguousarray(pixels, dtype=np.float32)):
            raise IOError(f"OpenImageIO failed to write EXR file {file_path}: {out.geterror()}")
    finally:
        out.close()

def write_exr_like(file_path, source_path, rgb, alpha=None, keep=None):
    # Writes rgb (float32 [H, W, 3] at the source's display window size) into a copy of the
    # source EXR. Every other channel, layer, part and header attribute (compression, pixel
    # formats, metadata) is the source's. Where `keep` ([H, W] bool) is True the source
    # pixels are written back unchanged, so untouched areas stay bit-identical. The data
    # window is the source's, grown only to hold new non-zero pixels outside it.
    # Modelled on write_exr_like in AInVFX Paint a plate (Apache-2.0), rewritten for OpenImageIO.
    require_oiio()
    inp = oiio.ImageInput.open(source_path)
    if inp is None:
        raise ValueError(f"OpenImageIO failed to open source EXR {source_path}: {oiio.geterror()}")
    try:
        specs = []
        while inp.seek_subimage(len(specs), 0):
            specs.append(oiio.ImageSpec(inp.spec()))
        if any(s.deep for s in specs):
            raise ValueError(f"Deep EXRs are not supported as a source: {source_path}")
        target = next((i for i, s in enumerate(specs) if all(c in s.channelnames for c in "RGB")), None)
        if target is None:
            raise ValueError(f"Source EXR {source_path} has no R, G, B channels to replace")
        spec = specs[target]
        height, width = rgb.shape[:2]
        if (width, height) != (spec.full_width, spec.full_height):
            raise ValueError(f"Image size {width}x{height} does not match the source display window "
                             f"{spec.full_width}x{spec.full_height} of {source_path}")

        names = list(spec.channelnames)
        r, g, b = (names.index(c) for c in "RGB")
        a = names.index("A") if "A" in names else None
        edit = np.ones((height, width), bool) if keep is None else ~keep

        # Source data window in display-window coordinates (it may extend past them: overscan)
        x0, y0 = spec.x - spec.full_x, spec.y - spec.full_y
        x1, y1 = x0 + spec.width, y0 + spec.height
        inside = np.zeros((height, width), bool)
        inside[max(y0, 0):max(min(y1, height), 0), max(x0, 0):max(min(x1, width), 0)] = True
        visible = np.abs(rgb).max(axis=-1) > 0
        if alpha is not None:
            visible |= alpha > 0
        grow = edit & ~inside & visible
        if grow.any():
            ys, xs = np.nonzero(grow)
            x0, y0 = min(x0, int(xs.min())), min(y0, int(ys.min()))
            x1, y1 = max(x1, int(xs.max()) + 1), max(y1, int(ys.max()) + 1)

        roi = oiio.ROI(x0 + spec.full_x, x1 + spec.full_x, y0 + spec.full_y, y1 + spec.full_y, 0, 1, 0, spec.nchannels)
        buf = oiio.ImageBuf(source_path, target, 0)
        if buf.roi != roi:
            buf = oiio.ImageBufAlgo.crop(buf, roi)  # pads with zeros where the window grew
        px = buf.get_pixels(oiio.FLOAT, roi)
        if buf.has_error or px is None:
            raise ValueError(f"OpenImageIO failed to read {source_path}: {buf.geterror()}")
        px = px.reshape(y1 - y0, x1 - x0, spec.nchannels)

        out_spec = oiio.ImageSpec(spec)
        out_spec.x, out_spec.y = x0 + spec.full_x, y0 + spec.full_y
        out_spec.width, out_spec.height = x1 - x0, y1 - y0
        if alpha is not None and a is None:
            # Add A, starting from what Load EXR reports for a file without alpha:
            # opaque inside the source data window, transparent outside
            src_alpha = np.zeros(px.shape[:2], np.float32)
            src_alpha[spec.y - out_spec.y:spec.y - out_spec.y + spec.height,
                      spec.x - out_spec.x:spec.x - out_spec.x + spec.width] = 1.0
            px = np.dstack([px, src_alpha])
            a = len(names)
            names.append("A")
            formats = list(spec.channelformats)
            out_spec.nchannels = len(names)
            out_spec.channelnames = tuple(names)
            if formats:
                out_spec.channelformats = tuple(formats + [formats[r]])
            out_spec.alpha_channel = a

        # Where the display window and the written window overlap
        dx0, dy0 = max(x0, 0), max(y0, 0)
        dx1, dy1 = min(x1, width), min(y1, height)
        if dx1 > dx0 and dy1 > dy0:
            window = px[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
            change = edit[dy0:dy1, dx0:dx1]
            for k, c in enumerate((r, g, b)):
                window[..., c] = np.where(change, rgb[dy0:dy1, dx0:dx1, k], window[..., c])
            if alpha is not None:
                window[..., a] = np.where(change, alpha[dy0:dy1, dx0:dx1], window[..., a])

        specs[target] = out_spec
        out = oiio.ImageOutput.create(file_path)
        if out is None:
            raise IOError(f"OpenImageIO cannot create an EXR writer for {file_path}: {oiio.geterror()}")
        try:
            ok = out.open(file_path, specs) if len(specs) > 1 else out.open(file_path, out_spec)
            for i, s in enumerate(specs):
                if not ok:
                    break
                if i > 0:
                    ok = out.open(file_path, s, "AppendSubimage")
                if not ok:
                    break
                if i == target:
                    ok = out.write_image(np.ascontiguousarray(px, dtype=np.float32))
                else:
                    ok = inp.seek_subimage(i, 0) and out.copy_image(inp)
            if not ok:
                raise IOError(f"OpenImageIO failed to write EXR file {file_path}: {out.geterror()}")
        finally:
            out.close()
    finally:
        inp.close()

class LoadEXRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "C:/path/to/your/image.####.exr", "multiline": False}),
                # Auto reads the EXR `chromaticities` header: ACEScg and ACES2065-1 are converted to
                # linear sRGB, Rec.709 is kept, no attribute is taken as ACEScg
                "color_space": (["Raw (Keep ACEScg)", "Convert ACEScg to Linear sRGB", "Auto (from EXR header)"],),
            },
            # New widgets go last: saved UI workflows store widget values by position
            "optional": {
                "load_mode": (["single_frame", "image_sequence"], {"default": "single_frame"}),
                "frame_index": ("INT", {"default": 1, "min": 0, "max": 999999, "step": 1}),
                "sequence_limit": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "missing_frames": (["skip", "hold previous"], {
                    "default": "skip",
                    "tooltip": "Sequences with gaps: 'skip' loads only the frames on disk; 'hold previous' "
                               "repeats the last frame found, so batch index = frame - first frame."}),
            }
        }

    CATEGORY = "image/EXR"
    # alpha is straight EXR alpha (1 = opaque), all ones when the file has none.
    # Note this is the inverse of the MASK that ComfyUI's LoadImage outputs.
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "load_exr"

    @classmethod
    def IS_CHANGED(cls, image_path, color_space, load_mode="single_frame", frame_index=1, sequence_limit=0,
                   missing_frames="skip"):
        # Re-run when any file on disk changes, not only when the inputs change
        try:
            files = cls()._collect_files(image_path, load_mode, frame_index, sequence_limit, missing_frames)
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

    def _get_sequence_files(self, path_pattern, start_frame=0, limit=0, hold=False):
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
        if hold and matches:
            # Every frame number from the first to the last found; a gap repeats the frame before it
            found = dict(matches)
            filled, last = [], None
            for frame in range(matches[0][0], matches[-1][0] + 1):
                last = found.get(frame, last)
                filled.append((frame, last))
            matches = filled

        if limit > 0:
            matches = matches[:limit]

        return [m[1] for m in matches]

    def _collect_files(self, image_path, load_mode, frame_index, sequence_limit, missing_frames="skip"):
        image_path = image_path.strip('"').strip("'")
        is_padded = bool(re.search(r'#+|%\d*d', image_path))

        if load_mode == "image_sequence" and is_padded:
            files = self._get_sequence_files(image_path, start_frame=frame_index, limit=sequence_limit,
                                             hold=missing_frames == "hold previous")
            if not files:
                raise FileNotFoundError(f"No sequence found for pattern: {image_path} starting at frame {frame_index}")
            return files

        if load_mode == "single_frame" and is_padded:
            image_path = self._resolve_path(image_path, frame_index)
        return [self._resolve_input_path(image_path)]

    def load_exr(self, image_path, color_space, load_mode="single_frame", frame_index=1, sequence_limit=0,
                 missing_frames="skip"):
        files_to_load = self._collect_files(image_path, load_mode, frame_index, sequence_limit, missing_frames)

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
            elif color_space == "Auto (from EXR header)":
                primaries = exr_primaries(file_path)
                if primaries not in AUTO_TO_SRGB:
                    raise ValueError(f"{file_path} has chromaticities {primaries}, which are not ACEScg, "
                                     f"ACES2065-1 or Rec.709; choose Raw and convert with an OCIO node")
                if AUTO_TO_SRGB[primaries] is not None:
                    img = apply_color_matrix(img, AUTO_TO_SRGB[primaries])

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
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional source EXR (or name.####.exr pattern) to write like: keeps its other layers, "
                               "parts, header, pixel formats, compression and data window. Precision and "
                               "compression are then the source's."}),
                "source_frame": ("INT", {
                    "default": 1, "min": 0, "max": 999999,
                    "tooltip": "Frame of source_path for the first image of the batch (when it is a pattern)."}),
                # Where pixels may differ from the source; elsewhere the source pixels are kept bit-identical
                "changed": ("MASK", ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "image/EXR"

    def _source_file(self, source_path, frame):
        loader = LoadEXRNode()
        path = source_path.strip().strip('"').strip("'")
        if re.search(r'#+|%\d*d', path):
            path = loader._resolve_path(path, frame)
        path = loader._resolve_input_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source EXR not found: {path}")
        return path

    def save_exr(self, images, filename_prefix="ComfyUI_EXR", color_space="Raw (Keep Current)", alpha=None,
                 precision="32-bit float", compression="zip", source_path="", source_frame=1, changed=None,
                 prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        if alpha is not None and alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        if changed is not None and changed.dim() == 2:
            changed = changed.unsqueeze(0)
        like_source = bool(source_path and source_path.strip())
        chromaticities = "ACEScg" if color_space == "Convert Linear sRGB to ACEScg" else None
        pixel_format = "half" if precision == "16-bit half" else "float"

        results = list()

        for batch_index, image in enumerate(images):
            img_np = image.cpu().numpy().astype(np.float32)

            if color_space == "Convert Linear sRGB to ACEScg":
                img_np = apply_color_matrix(img_np, SRGB_TO_ACESCG_MATRIX)

            alpha_np = None
            if alpha is not None:
                alpha_np = alpha[min(batch_index, alpha.shape[0] - 1)].cpu().numpy().astype(np.float32)
                if alpha_np.shape != img_np.shape[:2]:
                    raise ValueError(
                        f"Alpha size {alpha_np.shape[1]}x{alpha_np.shape[0]} does not match "
                        f"image size {img_np.shape[1]}x{img_np.shape[0]}"
                    )

            # Plain zero padding: Nuke/Resolve read it as a name_#####.exr sequence, and
            # ComfyUI's counter scan parses it (the old "0_001" form made every run restart at 1)
            file_name = f"{filename}_{counter:05}.exr"
            file_path = os.path.join(full_output_folder, file_name)

            if like_source:
                keep = None
                if changed is not None:
                    keep = changed[min(batch_index, changed.shape[0] - 1)].cpu().numpy() <= 1.0 / 1024
                    if keep.shape != img_np.shape[:2]:
                        raise ValueError(f"'changed' size {keep.shape[1]}x{keep.shape[0]} does not match "
                                         f"image size {img_np.shape[1]}x{img_np.shape[0]}")
                write_exr_like(file_path, self._source_file(source_path, source_frame + batch_index),
                               img_np, alpha_np, keep)
            else:
                if alpha_np is not None:
                    img_np = np.dstack([img_np, alpha_np])
                write_exr(file_path, img_np, pixel_format, compression, chromaticities)

            results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
