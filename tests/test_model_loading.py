# SPDX-License-Identifier: Apache-2.0
import json

from omlx.utils.model_loading import prepare_model_for_loading


def test_prepare_model_for_loading_adds_quantization_for_jang(tmp_path):
    model_dir = tmp_path / "jang-model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "quantization_config": {
                    "quant_method": "jang2",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        ),
        encoding="utf-8",
    )

    prepare_model_for_loading(str(model_dir))

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["quantization"] == {"bits": 4, "group_size": 128, "mode": "affine"}


def test_prepare_model_for_loading_does_not_override_existing_quantization(tmp_path):
    model_dir = tmp_path / "already-quant"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "quantization": {"bits": 6, "group_size": 64, "mode": "mxfp4"},
                "quantization_config": {"quant_method": "jang2", "bits": 4},
            }
        ),
        encoding="utf-8",
    )

    prepare_model_for_loading(str(model_dir))

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["quantization"] == {"bits": 6, "group_size": 64, "mode": "mxfp4"}
