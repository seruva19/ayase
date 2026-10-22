# ViSQOL backend

Ayase runs ViSQOL through the pinned `visqol-python` 3.8.0 package by default. The
backend identity recorded on results is `visqol_python_port`. It is a NumPy/SciPy
transcription of Google's published ViSQOL v3 protocol and bundles the official
audio SVR and speech deep-lattice TFLite models.

Audio mode uses the official 48 kHz SVR mapping. Speech mode uses the 16 kHz
deep-lattice mapper and fails setup if the lattice runtime or model cannot load;
Ayase never silently substitutes the polynomial speech mapping.

The upstream port reports conformance against the official C++ v3.3.3 golden set:
the maximum reported absolute MOS difference is 0.000117 for audio and 0.002341
for lattice speech. Ayase treats the port as the published metric because it keeps
the protocol, models, search ranges, alignment stages, and FP64 scoring path. The
backend identity remains explicit so results can distinguish it from Google's
native binding and CLI.

Ayase independently reran the port's 12 pinned conformance cases on Windows with
`visqol-python==3.8.0`, `libsvm-official==3.37.0`, and
`ai-edge-litert==2.3.0`; all 12 passed against checksum-verified official WAV data.

Google's native Python binding remains available as `visqol_python_official`, and
an installed CLI remains available as `visqol_cli`. Native bindings load WAV data
as arrays before calling `Measure` and resolve model files relative to the binding
package. Set `visqol_executable` to select a non-default CLI path.

Sources:

- ViSQOL v3 paper and implementation: https://github.com/google/visqol
- Python transcription and conformance table: https://github.com/talker93/visqol-python
