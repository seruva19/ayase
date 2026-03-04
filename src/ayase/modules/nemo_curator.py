"""NeMo Curator Quality module.

Scores the quality of caption text associated with a sample using a
three-tier backend:

  1. **DeBERTa quality classifier** — ``nvidia/quality-classifier-deberta``
     from HuggingFace, the same model used inside NeMo Curator's
     ``QualityClassifier`` pipeline stage.  Trained on 22 828 Common Crawl
     samples labelled Low / Medium / High.  Note: the model was trained on
     *web documents*, not image/video captions specifically, but transfers
     well to caption quality assessment.
     Requires ``pip install transformers torch huggingface_hub``.
  2. **FastText** quality filter — ``pip install fasttext``.
     Requires a pre-trained quality model file (set ``fasttext_model`` in config).
  3. **Heuristic** — always works; checks length, punctuation ratio,
     capitalisation, repetition, and special-character ratio.

Only processes samples that have ``sample.caption.text``.  Stores
``nemo_quality_score`` (0–1) and ``nemo_quality_label``
(Low / Medium / High) on the sample's QualityMetrics.
"""

import logging
import re
from typing import Optional, Tuple

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_DEBERTA_MODEL_ID = "nvidia/quality-classifier-deberta"

# Lazy-initialised module-level class so it's picklable and only built once.
_QualityModel = None


def _get_quality_model_class():
    """Build the QualityModel class on first use (avoids import-time torch dep)."""
    global _QualityModel
    if _QualityModel is not None:
        return _QualityModel

    import torch
    from torch import nn
    from transformers import AutoModel
    from huggingface_hub import PyTorchModelHubMixin

    class QualityModel(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config):
            super().__init__()
            self.model = AutoModel.from_pretrained(config["base_model"])
            self.dropout = nn.Dropout(config["fc_dropout"])
            self.fc = nn.Linear(
                self.model.config.hidden_size, len(config["id2label"])
            )

        def forward(self, input_ids, attention_mask):
            features = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            dropped = self.dropout(features)
            outputs = self.fc(dropped)
            return torch.softmax(outputs[:, 0, :], dim=1)

    _QualityModel = QualityModel
    return _QualityModel


class NemoCuratorModule(PipelineModule):
    name = "nemo_curator"
    description = "Caption text quality scoring (DeBERTa/FastText/heuristic)"
    default_config = {
        "backend": "auto",  # auto | deberta | fasttext | heuristic
        "model_name": _DEBERTA_MODEL_ID,
        "min_length": 10,
        "max_length": 2000,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend: Optional[str] = None
        self._deberta_model = None
        self._deberta_tokenizer = None
        self._deberta_config = None
        self._deberta_device = "cpu"
        self._fasttext_model = None
        self.min_length = self.config.get("min_length", 10)
        self.max_length = self.config.get("max_length", 2000)

    def setup(self) -> None:
        preferred = self.config.get("backend", "auto")

        # Tier 1: DeBERTa quality classifier (nvidia/quality-classifier-deberta)
        # Uses a custom QualityModel architecture (DeBERTa V3 Base + linear head)
        # loaded via PyTorchModelHubMixin, NOT AutoModelForSequenceClassification.
        if preferred in ("auto", "deberta"):
            try:
                import torch
                from transformers import AutoConfig, AutoTokenizer

                model_name = self.config.get("model_name", _DEBERTA_MODEL_ID)
                device = "cuda" if torch.cuda.is_available() else "cpu"

                QualityModel = _get_quality_model_class()

                self._deberta_config = AutoConfig.from_pretrained(model_name)
                self._deberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._deberta_model = QualityModel.from_pretrained(
                    model_name
                ).to(device)
                self._deberta_model.eval()
                self._deberta_device = device

                self._backend = "deberta"
                logger.info(f"NeMo Curator: DeBERTa classifier on {device}")
                return
            except ImportError:
                if preferred == "deberta":
                    logger.warning("transformers/torch/huggingface_hub not installed for DeBERTa backend")
            except Exception as e:
                logger.warning(f"DeBERTa classifier init failed: {e}")

        # Tier 2: FastText
        if preferred in ("auto", "fasttext"):
            try:
                import fasttext  # noqa: F401

                model_path = self.config.get("fasttext_model", None)
                if model_path:
                    import os
                    if os.path.exists(model_path):
                        self._fasttext_model = fasttext.load_model(model_path)
                        self._backend = "fasttext"
                        logger.info("NeMo Curator: using FastText backend")
                        return
                # No model file → fall through
                if preferred == "fasttext":
                    logger.warning("FastText model file not found")
            except ImportError:
                if preferred == "fasttext":
                    logger.warning("FastText not installed")
            except Exception as e:
                logger.warning(f"FastText init failed: {e}")

        # Tier 3: Heuristic (always works)
        self._backend = "heuristic"
        logger.info("NeMo Curator: using heuristic backend")

    def process(self, sample: Sample) -> Sample:
        if not sample.caption or not sample.caption.text:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        text = sample.caption.text

        try:
            score, label = self._score_text(text)
            sample.quality_metrics.nemo_quality_score = score
            sample.quality_metrics.nemo_quality_label = label
        except Exception as e:
            logger.warning(f"NeMo Curator failed for {sample.path}: {e}")

        return sample

    def _score_text(self, text: str) -> Tuple[float, str]:
        if self._backend is None:
            logger.warning("NeMo Curator: setup() was not called, using heuristic fallback")
            self._backend = "heuristic"
        if self._backend == "deberta" and self._deberta_model is not None:
            return self._score_deberta(text)
        if self._backend == "fasttext" and self._fasttext_model is not None:
            return self._score_fasttext(text)
        return self._score_heuristic(text)

    # ── Tier 1: DeBERTa ─────────────────────────────────────────────

    def _score_deberta(self, text: str) -> Tuple[float, str]:
        try:
            import torch

            inputs = self._deberta_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="longest",
            ).to(self._deberta_device)

            with torch.no_grad():
                # Custom model takes positional args and returns softmax probs directly
                probs = self._deberta_model(
                    inputs["input_ids"], inputs["attention_mask"]
                )

            probs = probs.squeeze(0).cpu().numpy()
            id2label = self._deberta_config.id2label

            # Weighted score: map class probabilities to 0-1
            label_weights = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
            score = 0.0
            for idx, prob in enumerate(probs):
                label = id2label.get(idx, f"LABEL_{idx}")
                weight = label_weights.get(label, 0.5)
                score += float(prob) * weight

            score = max(0.0, min(1.0, score))

            # Best class
            best_idx = int(probs.argmax())
            best_label = id2label.get(best_idx, "Medium")

            return score, best_label
        except Exception as e:
            logger.debug(f"DeBERTa scoring failed, falling back to heuristic: {e}")
            return self._score_heuristic(text)

    # ── Tier 2: FastText ─────────────────────────────────────────────

    def _score_fasttext(self, text: str) -> Tuple[float, str]:
        try:
            prediction = self._fasttext_model.predict(text.replace("\n", " "))
            label = prediction[0][0] if prediction[0] else "__label__medium"
            conf = float(prediction[1][0]) if prediction[1] else 0.5

            label_map = {
                "__label__high": 1.0,
                "__label__medium": 0.5,
                "__label__low": 0.0,
            }
            base = label_map.get(label, 0.5)
            score = base * conf + 0.5 * (1 - conf)
            return max(0.0, min(1.0, score)), self._label_from_score(score)
        except Exception:
            return self._score_heuristic(text)

    # ── Tier 3: Heuristic ────────────────────────────────────────────

    def _score_heuristic(self, text: str) -> Tuple[float, str]:
        score = 0.0
        total_weight = 0.0

        # Length check (weight 0.3)
        length = len(text)
        if length < self.min_length:
            len_score = length / max(self.min_length, 1)
        elif length > self.max_length:
            len_score = max(0.0, 1.0 - (length - self.max_length) / self.max_length)
        else:
            len_score = 1.0
        score += len_score * 0.3
        total_weight += 0.3

        # Punctuation ratio (weight 0.15)
        punct_count = sum(1 for c in text if c in ".,;:!?")
        word_count = max(len(text.split()), 1)
        punct_ratio = punct_count / word_count
        punct_score = 1.0 if 0.05 <= punct_ratio <= 0.4 else max(0.0, 1.0 - abs(punct_ratio - 0.2) * 3)
        score += punct_score * 0.15
        total_weight += 0.15

        # Capitalisation (weight 0.15)
        if text and text[0].isupper():
            cap_score = 1.0
        elif text and text[0].isalpha():
            cap_score = 0.3
        else:
            cap_score = 0.5
        score += cap_score * 0.15
        total_weight += 0.15

        # Repetition detection (weight 0.25)
        words = text.lower().split()
        if len(words) >= 3:
            unique_ratio = len(set(words)) / len(words)
            rep_score = min(unique_ratio * 1.2, 1.0)
        else:
            rep_score = 0.5
        score += rep_score * 0.25
        total_weight += 0.25

        # Special character ratio (weight 0.15)
        special_count = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?\'"()-]', text))
        special_ratio = special_count / max(len(text), 1)
        special_score = max(0.0, 1.0 - special_ratio * 10)
        score += special_score * 0.15
        total_weight += 0.15

        final = score / max(total_weight, 1e-6)
        final = max(0.0, min(1.0, final))
        return final, self._label_from_score(final)

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score >= 0.7:
            return "High"
        if score >= 0.4:
            return "Medium"
        return "Low"
