"""Tests for the EXR nodes, run against ComfyUI's real folder_paths.

Run with the ComfyUI venv (stdlib unittest, no pytest needed):
    <ComfyUI>/venv/Scripts/python.exe -m unittest discover -s tests -v

COMFYUI_ROOT points at the ComfyUI checkout (default L:/ComfyUI).
EXR_TEST_TMP sets where temp files go (default: system temp).
"""
import importlib.util
import os
import sys
import tempfile
import unittest

COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "L:/ComfyUI")
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFYUI_ROOT)

import numpy as np  # noqa: E402
import OpenImageIO as oiio  # noqa: E402
import torch  # noqa: E402
import folder_paths  # noqa: E402

spec = importlib.util.spec_from_file_location("exr_nodes", os.path.join(PACK_DIR, "exr_nodes.py"))
exr_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exr_nodes)


class ExrNodeTests(unittest.TestCase):
    def test_pack_registers_both_nodes(self):
        spec = importlib.util.spec_from_file_location(
            "exr_pack", os.path.join(PACK_DIR, "__init__.py"), submodule_search_locations=[PACK_DIR])
        pack = importlib.util.module_from_spec(spec)
        sys.modules["exr_pack"] = pack
        spec.loader.exec_module(pack)
        self.assertEqual(set(pack.NODE_CLASS_MAPPINGS), {"Load EXR (ACEScg)", "Save EXR (ACEScg)"})

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("EXR_TEST_TMP"))
        self.input_dir = os.path.join(self._tmp.name, "input")
        self.output_dir = os.path.join(self._tmp.name, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        folder_paths.set_input_directory(self.input_dir)
        folder_paths.set_output_directory(self.output_dir)
        self.load = exr_nodes.LoadEXRNode()
        self.save = exr_nodes.SaveEXRNode()

    def tearDown(self):
        self._tmp.cleanup()

    def write_exr(self, name, array, channels=None, fmt="float", origin=(0, 0), full=None, attrs=None, parts=()):
        # Fixture writer: array is [H, W, C] in file channel order, origin is the
        # data window offset, full=(w, h) the display window, attrs extra header
        # attributes and parts extra (name, array) subimages
        path = os.path.join(self.input_dir, name)

        def make_spec(array, channels, part_name=None):
            height, width, count = array.shape
            spec = oiio.ImageSpec(width, height, count, fmt)
            spec.channelnames = tuple(channels or ("R", "G", "B", "A", "Y")[:count] if count != 1 else ("Y",))
            spec.x, spec.y = origin
            if full:
                spec.full_x, spec.full_y = 0, 0
                spec.full_width, spec.full_height = full
            for key, value in (attrs or {}).items():
                if isinstance(value, tuple):
                    spec.attribute(key, oiio.TypeDesc(f"float[{len(value)}]"), value)
                else:
                    spec.attribute(key, value)
            if part_name:
                spec.attribute("name", part_name)
            return spec

        def as3d(a):
            a = np.asarray(a, dtype=np.float32)
            return a[:, :, None] if a.ndim == 2 else a

        array = as3d(array)
        images = [(make_spec(array, channels, "rgba" if parts else None), array)]
        images += [(make_spec(as3d(a), None if as3d(a).shape[2] != 1 else ("Z",), n), as3d(a)) for n, a in parts]
        out = oiio.ImageOutput.create(path)
        specs = [s for s, _ in images]
        self.assertTrue(out.open(path, specs) if len(specs) > 1 else out.open(path, specs[0]), oiio.geterror())
        for i, (spec, data) in enumerate(images):
            if i:
                self.assertTrue(out.open(path, spec, "AppendSubimage"), out.geterror())
            self.assertTrue(out.write_image(data), out.geterror())
        out.close()
        return path

    def read_raw(self, path):
        inp = oiio.ImageInput.open(path)
        spec, pixels = inp.spec(), inp.read_image(format="float")
        inp.close()
        return spec, pixels

    def test_save_names_are_nuke_sequence_friendly_and_do_not_overwrite(self):
        images = torch.rand(2, 4, 5, 3)
        self.save.save_exr(images, "shot/plate")
        self.save.save_exr(images[:1], "shot/plate")
        names = sorted(os.listdir(os.path.join(self.output_dir, "shot")))
        self.assertEqual(names, ["plate_00001.exr", "plate_00002.exr", "plate_00003.exr"])

    def test_rgb_roundtrip_is_lossless_and_keeps_hdr(self):
        images = torch.rand(1, 6, 7, 3) * 8 - 1  # values outside 0..1 must survive
        result = self.save.save_exr(images, "rt")
        name = result["ui"]["images"][0]["filename"]
        out_path = os.path.join(self.output_dir, name)
        image, alpha = self.load.load_exr(out_path, "Raw (Keep ACEScg)")
        self.assertEqual(tuple(image.shape), (1, 6, 7, 3))
        self.assertTrue(torch.allclose(image, images, atol=1e-6))
        self.assertTrue(torch.all(alpha == 1))

    def test_colorspace_roundtrip(self):
        images = torch.rand(1, 4, 4, 3) * 4
        result = self.save.save_exr(images, "cs", color_space="Convert Linear sRGB to ACEScg")
        out_path = os.path.join(self.output_dir, result["ui"]["images"][0]["filename"])
        acescg, _ = self.load.load_exr(out_path, "Raw (Keep ACEScg)")
        self.assertFalse(torch.allclose(acescg, images, atol=1e-3))
        back, _ = self.load.load_exr(out_path, "Convert ACEScg to Linear sRGB")
        self.assertTrue(torch.allclose(back, images, atol=1e-4))

    def test_alpha_is_loaded_and_saved(self):
        rgba = np.random.rand(5, 6, 4).astype(np.float32)
        path = self.write_exr("rgba.exr", rgba)
        image, alpha = self.load.load_exr(path, "Convert ACEScg to Linear sRGB")
        self.assertEqual(tuple(image.shape), (1, 5, 6, 3))
        self.assertTrue(np.allclose(alpha[0].numpy(), rgba[:, :, 3]))
        raw, _ = self.load.load_exr(path, "Raw (Keep ACEScg)")
        self.assertTrue(np.allclose(raw[0].numpy(), rgba[:, :, :3]))

        result = self.save.save_exr(raw, "rgba_out", alpha=alpha)
        spec, saved = self.read_raw(os.path.join(self.output_dir, result["ui"]["images"][0]["filename"]))
        self.assertEqual(tuple(spec.channelnames), ("R", "G", "B", "A"))
        self.assertTrue(np.allclose(saved, rgba, atol=1e-6))

    def test_half_precision_and_compression_options(self):
        images = torch.rand(1, 4, 4, 3)
        result = self.save.save_exr(images, "half", precision="16-bit half", compression="piz")
        spec, saved = self.read_raw(os.path.join(self.output_dir, result["ui"]["images"][0]["filename"]))
        self.assertEqual(str(spec.format), "half")
        self.assertEqual(spec.get_string_attribute("compression"), "piz")
        self.assertTrue(np.allclose(saved, images[0].numpy(), atol=1e-3))

    def test_half_float_input(self):
        rgb = np.random.rand(3, 3, 3).astype(np.float32) * 10
        path = self.write_exr("half_in.exr", rgb, fmt="half")
        image, _ = self.load.load_exr(path, "Raw (Keep ACEScg)")
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(np.allclose(image[0].numpy(), rgb, rtol=1e-3))

    def test_data_window_smaller_than_display_window_is_padded(self):
        # Nuke-style bbox: 2x2 of data at (3, 1) inside a 6x4 format
        path = self.write_exr("bbox.exr", np.full((2, 2, 3), 5.0), origin=(3, 1), full=(6, 4))
        image, alpha = self.load.load_exr(path, "Raw (Keep ACEScg)")
        self.assertEqual(tuple(image.shape), (1, 4, 6, 3))
        self.assertTrue(torch.all(image[0, 1:3, 3:5] == 5.0))
        self.assertEqual(float(image[0].sum()), 5.0 * 2 * 2 * 3)
        self.assertEqual(float(alpha[0].sum()), 4.0)
        self.assertTrue(torch.all(alpha[0, 1:3, 3:5] == 1.0))

    def test_overscan_is_cropped_to_display_window(self):
        data = np.arange(6 * 6 * 3, dtype=np.float32).reshape(6, 6, 3)
        path = self.write_exr("overscan.exr", data, origin=(-1, -1), full=(4, 4))
        image, alpha = self.load.load_exr(path, "Raw (Keep ACEScg)")
        self.assertEqual(tuple(image.shape), (1, 4, 4, 3))
        self.assertTrue(np.allclose(image[0].numpy(), data[1:5, 1:5]))
        self.assertTrue(torch.all(alpha == 1))

    def test_multilayer_picks_main_rgba_by_name(self):
        planes = np.random.rand(2, 2, 5).astype(np.float32)
        path = self.write_exr("layers.exr", planes, channels=("diffuse.R", "A", "B", "G", "R"))
        image, alpha = self.load.load_exr(path, "Raw (Keep ACEScg)")
        self.assertTrue(np.allclose(image[0].numpy(), planes[:, :, [4, 3, 2]]))
        self.assertTrue(np.allclose(alpha[0].numpy(), planes[:, :, 1]))

    def test_single_channel_exr(self):
        path = self.write_exr("gray.exr", np.random.rand(3, 4) * 100)
        image, alpha = self.load.load_exr(path, "Raw (Keep ACEScg)")
        self.assertEqual(tuple(image.shape), (1, 3, 4, 3))
        self.assertTrue(torch.equal(image[..., 0], image[..., 2]))

    def test_sequence_patterns_and_relative_paths(self):
        for frame in (1001, 1002, 1003):
            self.write_exr(f"seq.{frame}.exr", np.full((2, 2, 3), frame, np.float32))
        image, _ = self.load.load_exr("seq.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1002, 0)
        self.assertEqual(image.shape[0], 2)
        self.assertEqual(float(image[0, 0, 0, 0]), 1002)
        image, _ = self.load.load_exr("seq.%04d.exr", "Raw (Keep ACEScg)", "single_frame", 1003)
        self.assertEqual(float(image[0, 0, 0, 0]), 1003)

    def test_sequence_size_mismatch_has_clear_error(self):
        self.write_exr("mix.0001.exr", np.zeros((2, 2, 3)))
        self.write_exr("mix.0002.exr", np.zeros((3, 2, 3)))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.load.load_exr("mix.####.exr", "Raw (Keep ACEScg)", "image_sequence", 0, 0)

    def saved_path(self, result, index=0):
        return os.path.join(self.output_dir, result["ui"]["images"][index]["filename"])

    def read_part(self, path, subimage=0):
        inp = oiio.ImageInput.open(path)
        self.assertTrue(inp.seek_subimage(subimage, 0))
        spec = oiio.ImageSpec(inp.spec())
        native = inp.read_image(subimage, 0, 0, spec.nchannels, "unknown")
        inp.close()
        return spec, native

    def test_sequence_gaps_hold_previous_frame_when_asked(self):
        for frame in (1001, 1002, 1004):
            self.write_exr(f"gap.{frame}.exr", np.full((2, 2, 3), frame, np.float32))
        skip, _ = self.load.load_exr("gap.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1001, 0)
        self.assertEqual(skip[:, 0, 0, 0].tolist(), [1001, 1002, 1004])
        held, _ = self.load.load_exr("gap.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1001, 0, "hold previous")
        self.assertEqual(held[:, 0, 0, 0].tolist(), [1001, 1002, 1002, 1004])
        limited, _ = self.load.load_exr("gap.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1001, 3, "hold previous")
        self.assertEqual(limited.shape[0], 3)
        self.assertNotEqual(exr_nodes.LoadEXRNode.IS_CHANGED("gap.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1001, 0),
                            exr_nodes.LoadEXRNode.IS_CHANGED("gap.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1001, 0,
                                                              "hold previous"))

    def test_auto_color_space_reads_chromaticities(self):
        rgb = np.random.rand(2, 3, 3).astype(np.float32) * 3
        auto = "Auto (from EXR header)"
        ap1 = self.write_exr("ap1.exr", rgb, attrs={"chromaticities": exr_nodes.CHROMATICITIES["ACEScg"]})
        rec709 = self.write_exr("rec709.exr", rgb, attrs={"chromaticities": exr_nodes.CHROMATICITIES["Rec.709"]})
        ap0 = self.write_exr("ap0.exr", rgb, attrs={"chromaticities": exr_nodes.CHROMATICITIES["ACES2065-1"]})
        bare = self.write_exr("bare.exr", rgb)
        converted, _ = self.load.load_exr(ap1, "Convert ACEScg to Linear sRGB")
        self.assertTrue(torch.allclose(self.load.load_exr(ap1, auto)[0], converted))
        self.assertTrue(torch.allclose(self.load.load_exr(bare, auto)[0], converted))  # no attribute: ACEScg
        self.assertTrue(np.allclose(self.load.load_exr(rec709, auto)[0][0].numpy(), rgb))
        # AP0 -> AP1 -> sRGB equals converting through ACEScg explicitly
        via_ap1 = rgb.reshape(-1, 3) @ exr_nodes.AP0_TO_AP1_MATRIX.T @ exr_nodes.ACESCG_TO_SRGB_MATRIX.T
        self.assertTrue(np.allclose(self.load.load_exr(ap0, auto)[0][0].numpy().reshape(-1, 3), via_ap1, atol=1e-5))
        odd = self.write_exr("p3.exr", rgb, attrs={"chromaticities": (0.68, 0.32, 0.265, 0.69, 0.15, 0.06, 0.3127, 0.329)})
        with self.assertRaisesRegex(ValueError, "not ACEScg"):
            self.load.load_exr(odd, auto)

    def test_save_acescg_tags_chromaticities_for_auto_load(self):
        images = torch.rand(1, 3, 4, 3) * 2
        path = self.saved_path(self.save.save_exr(images, "tag", color_space="Convert Linear sRGB to ACEScg"))
        self.assertEqual(exr_nodes.exr_primaries(path), "ACEScg")
        back, _ = self.load.load_exr(path, "Auto (from EXR header)")
        self.assertTrue(torch.allclose(back, images, atol=1e-4))
        raw = self.saved_path(self.save.save_exr(images, "untagged"))
        self.assertIsNone(exr_nodes.exr_primaries(raw))

    def test_save_like_source_keeps_layers_header_and_untouched_pixels(self):
        rng = np.random.default_rng(1)
        planes = rng.random((4, 6, 7)).astype(np.float32) * 4  # R G B A diffuse.R diffuse.G depth.Z, half
        src = self.write_exr("like.exr", planes, fmt="half", origin=(2, 1), full=(10, 6),
                             channels=("R", "G", "B", "A", "diffuse.R", "diffuse.G", "depth.Z"),
                             attrs={"compression": "piz", "nuke/node_hash": "abc123", "framesPerSecond": 24.0})
        image, alpha = self.load.load_exr(src, "Raw (Keep ACEScg)")
        edited = image.clone()
        edited[0, 2:4, 3:5] = 7.5  # inside the data window
        changed = torch.zeros(1, 6, 10)
        changed[0, 2:4, 3:5] = 1
        out = self.saved_path(self.save.save_exr(edited, "like_out", source_path="like.exr", changed=changed))

        src_spec, src_px = self.read_part(src)
        out_spec, out_px = self.read_part(out)
        self.assertEqual(tuple(out_spec.channelnames), tuple(src_spec.channelnames))
        self.assertEqual((out_spec.x, out_spec.y, out_spec.width, out_spec.height), (2, 1, 6, 4))
        self.assertEqual((out_spec.full_width, out_spec.full_height), (10, 6))
        self.assertEqual(str(out_spec.format), "half")
        self.assertEqual(out_spec.get_string_attribute("compression"), "piz")
        self.assertEqual(out_spec.get_string_attribute("nuke/node_hash"), "abc123")
        self.assertEqual(out_spec.get_float_attribute("framesPerSecond"), 24.0)
        # Edit lands in R, G, B only; data-window coords = display coords - origin (2, 1)
        self.assertTrue(np.all(out_px[1:3, 1:3, :3] == 7.5))
        mask = np.ones(out_px.shape[:2], bool)
        mask[1:3, 1:3] = False
        self.assertEqual(out_px[mask].tobytes(), src_px[mask].tobytes())  # untouched: bit-identical
        self.assertEqual(out_px[..., 3:].tobytes(), src_px[..., 3:].tobytes())  # alpha and other layers

    def test_save_like_source_grows_data_window_and_adds_alpha(self):
        src = self.write_exr("small.exr", np.ones((2, 2, 3), np.float32), origin=(3, 1), full=(8, 5))
        image, alpha = self.load.load_exr(src, "Raw (Keep ACEScg)")
        image[0, 4, 7] = 2.0  # outside the data window
        out = self.saved_path(self.save.save_exr(image, "grown", alpha=alpha, source_path=src))
        spec, px = self.read_part(out)
        self.assertEqual((spec.x, spec.y, spec.x + spec.width, spec.y + spec.height), (3, 1, 8, 5))
        self.assertEqual(tuple(spec.channelnames), ("R", "G", "B", "A"))
        self.assertEqual(float(px[4 - 1, 7 - 3, 0]), 2.0)
        back, back_alpha = self.load.load_exr(out, "Raw (Keep ACEScg)")
        self.assertTrue(torch.equal(back_alpha, alpha))
        self.assertTrue(torch.equal(back, image))

    def test_save_like_source_copies_other_parts_and_maps_sequence_frames(self):
        for frame in (1001, 1002):
            self.write_exr(f"mp.{frame}.exr", np.full((3, 3, 3), frame, np.float32),
                           parts=[("depth", np.full((3, 3), frame / 10, np.float32))])
        image, _ = self.load.load_exr("mp.####.exr", "Raw (Keep ACEScg)", "image_sequence", 1001, 0)
        result = self.save.save_exr(image + 1, "mp_out", source_path="mp.####.exr", source_frame=1001)
        for index, frame in enumerate((1001, 1002)):
            out = self.saved_path(result, index)
            _, rgb = self.read_part(out, 0)
            self.assertTrue(np.all(rgb == frame + 1))
            depth_spec, depth = self.read_part(out, 1)
            self.assertEqual(depth_spec.get_string_attribute("name"), "depth")
            self.assertTrue(np.allclose(depth, frame / 10))
        with self.assertRaises(FileNotFoundError):
            self.save.save_exr(image, "mp_missing", source_path="mp.####.exr", source_frame=2001)

    def test_is_changed_tracks_file_modification(self):
        path = self.write_exr("live.exr", np.zeros((2, 2, 3)))
        before = exr_nodes.LoadEXRNode.IS_CHANGED(path, "Raw (Keep ACEScg)")
        os.utime(path, (1, 1))
        after = exr_nodes.LoadEXRNode.IS_CHANGED(path, "Raw (Keep ACEScg)")
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
