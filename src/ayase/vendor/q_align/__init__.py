"""Vendored mPLUG-Owl2 / Q-Align architecture (q-future/one-align, ICML 2024).
Upstream code carried two breakages against current transformers: an
`icecream` dev-dependency import and a missing `Cache` import. Both are
fixed in-tree here so the toolkit needs no extra dependency and no
`trust_remote_code` download — weights still load from q-future/one-align.
"""

from .modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM
from .configuration_mplug_owl2 import MPLUGOwl2Config
