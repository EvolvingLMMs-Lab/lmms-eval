from .llava import Llava
from .otterhd import OtterHD
from .qwen_vl import Qwen_VL
from .fuyu import Fuyu
from .gpt4v import GPT4V
from .instructblip import InstructBLIP
from .minicpm_v import MiniCPM_V
import os

try:
    # enabling faster model download
    import hf_transfer

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass
