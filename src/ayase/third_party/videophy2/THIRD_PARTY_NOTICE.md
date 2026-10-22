# VideoPhy-2 AutoEval runtime notice

The Python files under `mplug_owl_video/` were copied without functional changes from
`Hritikbansal/videophy`, directory `VIDEOPHY2/mplug_owl_video`, at Git revision
`01d366c979e4a82875852adbf427ccc036e7373a`.

The upstream repository is distributed under the MIT license reproduced in `LICENSE`.
The individual mPLUG-Owl configuration, modeling, processing, and tokenization files retain
their original copyright and Apache License 2.0 headers. The complete Apache License 2.0 text
is reproduced in `LICENSE.apache-2.0`.

Ayase replaces the upstream package `__init__.py` with a minimal namespace initializer. This
avoids eagerly importing optional training components; inference imports the unchanged runtime
modules directly. No model weights are included. The official checkpoint is resolved separately
from `videophysics/videophy_2_auto` at revision
`ca38370e1dcc79c99c7017dd8755d0a087299afc`.
