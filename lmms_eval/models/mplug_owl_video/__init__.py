# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_mplug_owl": ["MPLUG_OWL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MplugOwlConfig"],
    "processing_mplug_owl": ["MplugOwlImageProcessor", "MplugOwlProcessor"],
    "tokenization_mplug_owl": ["MplugOwlTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mplug_owl"] = [
        "MPLUG_OWL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MplugOwlForConditionalGeneration",
        "MplugOwlModel",
    ]


if TYPE_CHECKING:
    from .configuration_mplug_owl import MPLUG_OWL_PRETRAINED_CONFIG_ARCHIVE_MAP, MplugOwlConfig
    from .tokenization_mplug_owl import MplugOwlTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mplug_owl import (
            MPLUG_OWL_PRETRAINED_MODEL_ARCHIVE_LIST,
            MplugOwlForConditionalGeneration,
            MplugOwlModel,
            MplugOwlPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

from .configuration_mplug_owl import *
from .modeling_mplug_owl import *
from .processing_mplug_owl import *
from .tokenization_mplug_owl import *
