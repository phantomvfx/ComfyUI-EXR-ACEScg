import os
import re

# CRITICAL FIX: This environment variable MUST be set BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import numpy as np
import folder_paths

# ACEScg (AP1) to Linear sRGB Matrix
ACESCG_TO_SRGB_MATRIX = np.array([
    [ 1.7050796, -0.6218677, -0.0832119],
    [-0.1302559,  1.1408028, -0.0105469],
    [-0.0240075, -0.1289676,  1.1529751]
], dtype=np.float32)

# Linear sRGB to ACEScg (AP1) Matrix
SRGB_TO_ACESCG_MATRIX = np.linalg.inv(ACESCG_TO_SRGB_MATRIX)

def apply_color_matrix(image_np, matrix):
    flat_img = image_np.reshape(-1, 3)
    converted = np.dot(flat_img, matrix.T)
    return converted.reshape(image_np.shape)

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
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_exr"

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

    def load_exr(self, image_path, color_space, load_mode="single_frame", frame_index=1, sequence_limit=0):
        image_path = image_path.strip('"').strip("'")
        
        is_padded = bool(re.search(r'#+|%\d*d', image_path))
        files_to_load = []

        if load_mode == "single_frame":
            if is_padded:
                target_path = self._resolve_path(image_path, frame_index)
            else:
                target_path = image_path
            
            if not os.path.isabs(target_path):
                input_dir = folder_paths.get_input_directory()
                potential_path = os.path.join(input_dir, target_path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            
            files_to_load.append(target_path)

        elif load_mode == "image_sequence":
            if is_padded:
                files_to_load = self._get_sequence_files(image_path, start_frame=frame_index, limit=sequence_limit)
                if not files_to_load:
                    raise FileNotFoundError(f"No sequence found for pattern: {image_path} starting at frame {frame_index}")
            else:
                target_path = image_path
                if not os.path.isabs(target_path):
                    input_dir = folder_paths.get_input_directory()
                    potential_path = os.path.join(input_dir, target_path)
                    if os.path.exists(potential_path):
                        target_path = potential_path
                files_to_load.append(target_path)
                
        tensors = []
        for file_path in files_to_load:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"EXR file not found at path: {file_path}")

            # Read EXR as 32-bit float
            img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise ValueError(f"OpenCV failed to load EXR image. Make sure it is a valid EXR file: {file_path}")

            # OpenCV loads as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply Color Space Conversion if requested
            if color_space == "Convert ACEScg to Linear sRGB":
                img = apply_color_matrix(img, ACESCG_TO_SRGB_MATRIX)

            # Convert to ComfyUI standard Tensor [1, H, W, C]
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            tensors.append(img_tensor)
            
        if not tensors:
            raise ValueError("No images loaded.")
            
        # Concatenate on batch dimension
        batch_tensor = torch.cat(tensors, dim=0)
        return (batch_tensor,)


# --- (Keep your SaveEXRNode down here exactly as it was) ---
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
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "image/EXR"

    def save_exr(self, images, filename_prefix="ComfyUI_EXR", color_space="Raw (Keep Current)", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        results = list()
        
        for image in images:
            img_np = image.cpu().numpy()

            if color_space == "Convert Linear sRGB to ACEScg":
                img_np = apply_color_matrix(img_np, SRGB_TO_ACESCG_MATRIX)

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = img_bgr.astype(np.float32)

            file_math = f"{filename}_{counter:05_}.exr"
            file_path = os.path.join(full_output_folder, file_math)
            
            cv2.imwrite(file_path, img_bgr)
            
            results.append({
                "filename": file_math,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }