from .exr_nodes import LoadEXRNode, SaveEXRNode

NODE_CLASS_MAPPINGS = {
    "Load EXR (ACEScg)": LoadEXRNode,
    "Save EXR (ACEScg)": SaveEXRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load EXR (ACEScg)": "Load EXR (ACEScg)",
    "Save EXR (ACEScg)": "Save EXR (ACEScg)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']