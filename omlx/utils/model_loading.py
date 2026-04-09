# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_JANG_METHODS = {"jang", "jang2"}


def prepare_model_for_loading(model_name: str) -> str:
    """Apply compatibility shims before mlx-lm loads a model.

    Currently adds a compatibility shim for JANG/JANG2-quantized local models:
    if ``config.json`` contains ``quantization_config.quant_method`` as jang/jang2
    but misses MLX's expected top-level ``quantization`` field, inject a
    minimal equivalent ``quantization`` entry so mlx-lm can detect quantized
    weights correctly.

    Args:
        model_name: Local model path or HF repo id.

    Returns:
        Path/id to pass to mlx-lm (currently unchanged).
    """
    model_path = Path(model_name)
    if not model_path.is_dir():
        return model_name

    config_path = model_path / "config.json"
    if not config_path.exists():
        return model_name

    try:
        import json

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return model_name

    if not isinstance(config, dict) or "quantization" in config:
        return model_name

    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        return model_name

    method = str(qconfig.get("quant_method", "")).lower()
    if method not in _JANG_METHODS:
        return model_name

    bits = qconfig.get("bits", qconfig.get("w_bit", qconfig.get("nbits", 4)))
    group_size = qconfig.get("group_size", qconfig.get("q_group_size", 64))
    mode = qconfig.get("mode", "affine")

    try:
        bits = int(bits)
    except Exception:
        bits = 4

    try:
        group_size = int(group_size)
    except Exception:
        group_size = 64

    config["quantization"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info(
        "Applied JANG quantization compatibility shim: %s (bits=%s, group_size=%s)",
        model_path,
        bits,
        group_size,
    )
    return model_name


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    from mlx_lm import load

    model_name = prepare_model_for_loading(model_name)
    return load(model_name, tokenizer_config=tokenizer_config)


def apply_post_load_transforms(model: Any, model_settings: Any = None) -> Any:
    """Apply optional post-load model transforms based on settings.

    Currently supports:
    - IndexCache: skip redundant indexer computation in DSA layers
    - GatedDeltaNet advance: fix missing cache.advance() in qwen3_5

    Args:
        model: A loaded mlx-lm model instance.
        model_settings: A ModelSettings instance (or None).

    Returns:
        The (possibly patched) model.
    """
    # GatedDeltaNet advance patch: always applied for qwen3_5 models
    # (no settings needed — auto-detected by model type)
    from ..patches.gated_delta_advance import apply_gated_delta_advance_patch

    if apply_gated_delta_advance_patch(model):
        logger.info("GatedDeltaNet advance() patch applied")

    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, index_cache_freq)
        if applied:
            logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model
