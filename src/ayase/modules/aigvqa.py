"""AIGVQA --- Multi-Dimensional AI-Generated VQA (ICCVW 2025).

GitHub: https://github.com/IntMeGroup/AIGVQA
Weights: https://huggingface.co/IntMeGroup/ICCVW_mos0_8B (8B, InternVL2-based)

AIGVQA is the SJTU-IntMeGroup entry to the VQualA 2025 GenAI-Bench AIGC video
quality challenge. It is not a stock InternVL chat model: the released
checkpoint is a custom two-stream regression network whose forward returns a
predicted MOS (``score1``) --- it does NOT generate a rateable text answer at
inference, and it cannot be loaded by ``transformers`` alone (see REVIVAL NOTES).
No installable/self-contained AIGVQA backend exists, and a CLIP multi-prompt
proxy is not AIGVQA, so nothing is emitted under the AIGVQA name until a real
backend is wired in. This module reports itself unavailable and leaves
``aigvqa_score`` unset.

Output field: ``aigvqa_score`` (populated only with a real backend).

REVIVAL NOTES (provisional --- no turnkey / self-contained backend)
Metric: AIGVQA / Overall Quality Predictor (VQualA 2025 @ ICCVW, SJTU-IntMeGroup).
Category: MULTI-CHECKPOINT LMM REGRESSION (not pip-installable, not stock InternVL).
Why provisional: The HF checkpoint ``IntMeGroup/ICCVW_mos0_8B`` ships weights
  (top-level modules ``vision_model``, ``language_model``, ``mlp1``, ``fast_mlp``,
  ``mlpscore``, ``evaluator``) but NO modeling code --- ``config.json`` auto_map
  points at ``modeling_internvl_chat.InternVLChatModel`` /
  ``configuration_internvl_chat.py`` which are ABSENT from the repo, so
  ``AutoModel.from_pretrained(..., trust_remote_code=True)`` fails. The custom
  architecture (a FAST-VQA fragment swin branch ``evaluator.fragments_backbone``
  + ``fast_mlp`` fused with the InternVL2 stream, ``mlpscore`` head emitting the
  regression MOS ``score1``) lives only in the GitHub repo and additionally needs
  a separate LOVE temporal checkpoint (``anonymousdb/LOVE-pretrain/temporal.pth``)
  and a two-stream (dynamic-tile + spatial-fragment) preprocessor. This module is
  constrained to torch/transformers only and cannot bundle that repo pipeline, so
  no faithful score is reproducible in-process. Verified against workbox: env has
  8xH100 but the checkpoint still requires the repo code + temporal.pth to run,
  which is outside the single-file module contract.
To revive (verified inference protocol, reverse-engineered from the repo):
  1. Clone https://github.com/IntMeGroup/AIGVQA and use ``AIGVQA_8B/`` (its
     ``model/internvl_chat_st2/`` custom InternVL + ``swin_backbone.py``).
  2. Env: python 3.9, ``pip install -r requirements.txt``, ``flash-attn==2.3.6``,
     decord, timm, deepspeed. Download ``IntMeGroup/ICCVW_mos0_8B`` (~16GB),
     ``IntMeGroup/ICCVW_mos0_st222``, and ``anonymousdb/LOVE-pretrain/temporal.pth``.
  3. Entry point: ``AIGVQA_8B/train/stage2_eval_AIGV.py`` (driven by
     ``shell/eval_score_overall1.sh``): ``--model_name_or_path IntMeGroup/ICCVW_mos0_8B
     --conv_style internlm2-chat --force_image_size 448 --max_dynamic_patch 6``.
  4. Per video the dataset builds TWO pixel streams --- InternVL dynamic tiles
     (``pixel_values``) + FAST-VQA spatial fragments (``pixel_values2``, 8 segments)
     --- and the fixed prompt: "How would you rate the overall quality of the
     video? Considering the Aesthetic Quality, Image Quality, Temporal Quality and
     Text-Video Alignment of this video and its prompt? prompt: <caption>.".
  5. ``score1 = model(mos=..., pixel_values=..., pixel_values2=..., input_ids=...,
     image_flags=..., labels=...)['score1'].item()`` is the predicted MOS, trained
     on a /100 scale (multiply by 100 for the challenge CSV ``Overall_MOS``). The
     challenge Track-I overall is a 0.25-weighted ensemble of two 8B + two 26B
     checkpoints; a single 8B (``mos0``) already yields a usable score.
  6. Validate you reproduce the repo's SRCC/PLCC on GenAI-Bench before flipping
     ``provisional=False``, and wire ``_backend="real"`` only once the checkpoint
     runs end-to-end from this module.
Source: github.com/IntMeGroup/AIGVQA (AIGVQA_8B/train/stage2_eval_AIGV.py,
  shell/eval_score_overall1.sh, data/final_test_mos0.jsonl); HF config.json +
  model.safetensors.index.json for IntMeGroup/ICCVW_mos0_8B; VQualA 2025 Challenge
  paper (ICCVW 2025).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AIGVQAModule(PipelineModule):
    name = "aigvqa"
    provisional = True  # no turnkey / self-contained real backend (see REVIVAL NOTES)
    description = "AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "aigvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = "unavailable"
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # The published AIGVQA checkpoint (IntMeGroup/ICCVW_mos0_8B) is a custom
        # two-stream InternVL2 regression model whose modeling code is NOT shipped
        # with the weights; it needs the GitHub repo pipeline + a separate LOVE
        # temporal.pth (see REVIVAL NOTES). There is no importable/turnkey backend,
        # so the metric stays unset rather than falling back to a proxy.
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "AIGVQA unavailable: IntMeGroup/ICCVW_mos0_8B has no self-contained "
            "loader (custom repo architecture + temporal.pth required); "
            "aigvqa_score will not be populated. See module REVIVAL NOTES."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        # Real-or-none: without the real AIGVQA backend, emit nothing.
        if not self._ml_available or self._backend != "real":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.aigvqa_score = float(score)
        except Exception as e:
            logger.warning("AIGVQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None
