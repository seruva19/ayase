# Bundled sources added during provenance cleanup

Ayase's MIT license applies to Ayase-authored code. Bundled upstream code and
model assets retain their own notices and terms. Backend availability and metric
provenance do not change those terms.

| Directory or file | Upstream | Retained notice |
|---|---|---|
| `cover/` | [COVER](https://github.com/taco-group/COVER) | `cover/LICENSE` (MIT) |
| `opens2v/` | [OpenS2V-Nexus](https://github.com/PKU-YuanGroup/OpenS2V-Nexus) | `opens2v/LICENSE` (Apache-2.0) |
| `qwen_image_bench/` | [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) | `qwen_image_bench/LICENSE` (Apache-2.0) |
| `tifascore/` | [TIFA](https://github.com/Yushi-Hu/tifa) | `tifascore/LICENSE` (Apache-2.0) |
| `videoalign/` | [VideoAlign](https://github.com/KlingAIResearch/VideoAlign) | `videoalign/LICENSE` (MIT) |
| `panns_cnn14.py` | [AudioLDM evaluation](https://github.com/haoheliu/audioldm_eval), [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) | `panns_notices/LICENSE`, `panns_notices/LICENSE.MIT` |
| `idreveal/` | [ID-Reveal](https://github.com/grip-unina/id-reveal), [POI-Forensics](https://github.com/grip-unina/poi-forensics) | `idreveal/LICENSE.txt` (informational and nonprofit use only) |
| `idreveal/TDDFA/` | [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) | `idreveal/TDDFA/LICENSE` (MIT); model asset restrictions below |
| `idreveal/retinaface/` | RetinaFace implementation bundled by upstream | `idreveal/retinaface/LICENSE.MIT` |

`idreveal/TDDFA/configs/bfm_noneck_v3.pkl` is a modified Basel Face Model
asset, restricted to academic use by its upstream notice. Commercial use
requires a separate license; see `idreveal/TDDFA/utils/bfm_readme.md` and the
[Basel Face Model terms](https://faces.dmi.unibas.ch/bfm/?nav=1-0&id=basel_face_model).
These restrictions also apply when a backend uses this asset indirectly.

`panns_cnn14.py` retains only the Cnn14 layers needed for logits and embeddings;
the upstream shell downloader is replaced by a caller-supplied checkpoint.
TIFA uses lazy imports for optional dependencies. Other integration modifications
remain visible in the bundled source and Ayase's Git history.

`provenance_notices.json` records the exact upstream revisions and SHA-256 hashes
of restored license notices. These are notice snapshots, not assertions that
every bundled implementation file matches those revisions byte for byte.

Other pre-existing bundled packages retain their notices within their own
directories. Model checkpoint terms are listed separately in `MODELS.md` and
in module `models` declarations where available.
