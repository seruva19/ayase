"""Contracts for the isolated backend dependency inventory."""

import re
from pathlib import Path

from packaging.requirements import Requirement

from ayase.backend_dependencies import (
    BACKEND_DEPENDENCIES,
    get_backend_dependency,
    managed_pip_options,
    managed_requirements,
)


def test_inventory_keys_match_module_names_and_are_unique():
    assert len(BACKEND_DEPENDENCIES) >= 30
    assert all(key == dependency.module for key, dependency in BACKEND_DEPENDENCIES.items())
    assert get_backend_dependency("audio_estoi") is BACKEND_DEPENDENCIES["audio_estoi"]
    assert get_backend_dependency("not_a_module") is None


def test_chd_provisions_the_native_titok_runtime():
    dependency = BACKEND_DEPENDENCIES["chd"]
    requirements = managed_requirements(dependency)
    assert dependency.python_version == "3.11"
    assert "torch==2.6.0+cu124" in requirements
    assert "huggingface-hub==0.36.0" in requirements
    assert "einops==0.8.1" in requirements
    assert "omegaconf==2.3.0" in requirements
    assert "safetensors==0.4.5" in requirements


def test_palate_provisions_native_dinov2_without_jax_or_transformer_substitution():
    dependency = BACKEND_DEPENDENCIES["palate"]
    requirements = managed_requirements(dependency)
    assert "torch==2.6.0+cu124" in requirements
    assert "torchvision==0.21.0+cu124" in requirements
    assert "omegaconf==2.3.0" in requirements
    assert not any(item.startswith(("jax", "xformers")) for item in requirements)


def test_avc_reward_provisions_native_omni_instead_of_a_manual_vllm_server():
    dependency = BACKEND_DEPENDENCIES["avc_reward"]
    requirements = managed_requirements(dependency)
    assert "transformers==4.57.6" in requirements
    assert "qwen-omni-utils==0.0.8" in requirements
    assert "av==15.1.0" in requirements
    assert "accelerate==1.12.0" in requirements
    assert dependency.conda_packages == ("ffmpeg=7.1.1",)
    assert not any(item.startswith(("vllm", "flash-attn")) for item in requirements)


def test_pypi_requirements_are_exactly_pinned():
    for dependency in BACKEND_DEPENDENCIES.values():
        for spec in dependency.requirements:
            if " @ https://" in spec:
                continue
            requirement = Requirement(spec)
            pins = [item for item in requirement.specifier if item.operator == "=="]
            assert len(pins) == 1, (dependency.module, spec)
            assert "*" not in pins[0].version, (dependency.module, spec)


def test_source_requirements_use_immutable_revisions():
    source_entries = [item for item in BACKEND_DEPENDENCIES.values() if item.source_url]
    assert {item.module for item in source_entries} == {
        "audio_utmos_v2",
        "camera_trajectory",
        "opens2v",
        "p1203",
        "ram_tagging",
    }
    for dependency in source_entries:
        assert dependency.source_url.startswith("https://github.com/")
        assert re.fullmatch(r"[0-9a-f]{40}", dependency.source_revision or "")
        if dependency.unsupported_reason:
            continue
        archive = f"{dependency.source_url}/archive/{dependency.source_revision}.zip"
        assert any(requirement.endswith(archive) for requirement in dependency.requirements)
        assert all(not requirement.startswith("git+") for requirement in dependency.requirements)


def test_every_backend_uses_managed_python_311():
    assert all(item.python_version == "3.11" for item in BACKEND_DEPENDENCIES.values())


def test_known_base_or_bundled_backends_are_not_provisioned():
    excluded = {
        "cdpam",  # base dependency
        "dover",  # bundled implementation
        "jedi_official",  # custom isolated worker
        "motion_smoothness",  # bundled RIFE implementation
    }
    assert excluded.isdisjoint(BACKEND_DEPENDENCIES)


def test_transnet_isolated_for_python39_compatible_base_install():
    dependency = BACKEND_DEPENDENCIES["scene_detection"]
    assert dependency.requirements == ("transnetv2-pytorch==1.0.5",)
    assert dependency.python_version == "3.11"


def test_videophy2_uses_the_upstream_transformers_runtime():
    dependency = BACKEND_DEPENDENCIES["videophy2_autoeval"]
    assert "transformers==4.28.1" in dependency.requirements
    assert "sentencepiece==0.1.99" in dependency.requirements


def test_syncreward_isolates_peav_and_supplies_ffmpeg_for_decoding():
    dependency = BACKEND_DEPENDENCIES["syncreward"]
    requirements = managed_requirements(dependency)
    assert "transformers==5.1.0" in requirements
    assert "torch==2.9.1+cu128" in requirements
    assert "torchaudio==2.9.1+cu128" in requirements
    assert "torchcodec==0.8.1" in requirements
    assert "ftfy==6.3.1" in requirements  # Ayase's compatibility imports load CLIP.
    assert "transformers==4.57.6" not in requirements
    assert "torch==2.6.0+cu124" not in requirements
    assert dependency.conda_packages == ("ffmpeg=7.1.1",)
    assert managed_pip_options(dependency)[-1].endswith("/cu128")


def test_geneval2_uses_the_published_qwen_runtime_in_managed_python():
    dependency = BACKEND_DEPENDENCIES["geneval2"]
    requirements = managed_requirements(dependency)
    assert dependency.python_version == "3.11"
    assert "transformers==4.57.0" in requirements
    assert "transformers==4.57.6" not in requirements
    assert "accelerate==1.12.0" in requirements
    assert "ftfy==6.3.1" in requirements


def test_morpheus_provisions_native_tracking_and_numerical_runtime():
    dependency = BACKEND_DEPENDENCIES["morpheus"]
    requirements = managed_requirements(dependency)
    assert dependency.python_version == "3.11"
    assert "torch==2.6.0+cu124" in requirements
    assert "transformers==4.57.6" in requirements
    assert "hydra-core==1.3.2" in requirements
    assert "iopath==0.1.10" in requirements
    assert "scikit-image==0.24.0" in requirements
    assert "ftfy==6.3.1" in requirements
    assert not dependency.unsupported_platforms


def test_anyaudio_provisions_omni_and_local_decomposition_runtime():
    requirements = managed_requirements(BACKEND_DEPENDENCIES["anyaudio_judge"])
    assert "torch==2.6.0+cu124" in requirements
    assert "transformers==4.57.6" in requirements
    assert "accelerate==1.12.0" in requirements
    assert "soundfile==0.12.1" in requirements
    assert "librosa==0.10.2.post1" in requirements
    assert "ftfy==6.3.1" in requirements


def test_deqa_isolates_the_native_legacy_stack():
    dependency = BACKEND_DEPENDENCIES["deqa_score"]
    requirements = managed_requirements(dependency)
    assert "torch==2.0.1+cu118" in requirements
    assert "transformers==4.36.1" in requirements
    assert "tokenizers==0.15.0" in requirements
    assert "protobuf==3.20.3" in requirements
    assert "icecream==2.1.3" in requirements
    assert "torch==2.6.0+cu124" not in requirements
    assert "transformers==4.57.6" not in requirements
    assert managed_pip_options(dependency)[-1].endswith("/cu118")


def test_edit_score_provisions_native_lora_and_parser_stack():
    requirements = managed_requirements(BACKEND_DEPENDENCIES["edit_score"])
    assert "torch==2.6.0+cu124" in requirements
    assert "transformers==4.57.6" in requirements
    assert "peft==0.17.1" in requirements
    assert "json-repair==0.30.3" in requirements
    assert "regex==2024.11.6" in requirements
    assert "ftfy==6.3.1" in requirements


def test_beyond_time_shifts_pins_native_omni_and_video_decoder():
    requirements = managed_requirements(BACKEND_DEPENDENCIES["beyond_time_shifts"])
    assert "transformers==4.57.1" in requirements
    assert "transformers==4.57.6" not in requirements
    assert "qwen-omni-utils==0.0.8" in requirements
    assert "av==15.1.0" in requirements
    assert "torchvision==0.21.0+cu124" in requirements
    assert "accelerate==1.12.0" in requirements
    assert "ftfy==6.3.1" in requirements


def test_native_binary_packages_are_declared_for_managed_conda_prefixes():
    assert BACKEND_DEPENDENCIES["audio_pesq"].conda_packages == ("pesq=0.0.4",)
    f0 = BACKEND_DEPENDENCIES["audio_log_f0_dtw"]
    assert f0.conda_packages == ("pyworld=0.3.5", "fastdtw=0.3.4")
    assert f0.requirements == ()  # exact NumPy SPTK recurrence is packaged with Ayase
    assert not f0.unsupported_platforms
    mcd = BACKEND_DEPENDENCIES["audio_mcd"]
    assert mcd.conda_packages == ("pyworld=0.3.5", "fastdtw=0.3.4")
    assert mcd.requirements == ()


def test_configured_alternative_backend_is_conditional_and_compatible():
    dependency = BACKEND_DEPENDENCIES["object_integrity"]
    assert dependency.config_conditions == (("backend", "mmpose"),)
    assert "mmcv==2.1.0" in dependency.requirements
    assert "mmcv==2.2.0" not in dependency.requirements
    assert dependency.pip_options[0] == "--find-links"
    assert "--no-build-isolation" in dependency.pip_options
    assert any("mattloper/chumpy/archive/580566e" in item for item in dependency.requirements)


def test_visqol_uses_managed_numerical_port_and_lattice_mapper():
    dependency = BACKEND_DEPENDENCIES["visqol"]
    assert "visqol-python==3.8.0" in dependency.requirements
    assert "ai-edge-litert==2.3.0" in dependency.requirements
    assert not dependency.unsupported_platforms
    assert "vmaf" not in BACKEND_DEPENDENCIES  # packaged FFmpeg supplies libvmaf
    assert not any(spec.startswith("torch") for spec in managed_requirements(dependency))


def test_managed_requirements_include_inference_runtime_without_duplicate_overrides():
    captioning = managed_requirements(BACKEND_DEPENDENCIES["captioning"])
    assert "torch==2.6.0+cu124" in captioning
    assert "transformers==4.57.6" in captioning
    assert "pycocoevalcap==1.2" in captioning

    videophy = managed_requirements(BACKEND_DEPENDENCIES["videophy2_autoeval"])
    assert "torch==2.0.1+cu118" in videophy
    assert "transformers==4.28.1" in videophy
    assert "torch==2.6.0+cu124" not in videophy
    assert "transformers==4.57.6" not in videophy
    assert managed_pip_options(BACKEND_DEPENDENCIES["captioning"])[-1].endswith("/cu124")
    assert any(
        option.endswith("/cu121")
        for option in managed_pip_options(BACKEND_DEPENDENCIES["object_integrity"])
    )


def test_cycle_reward_uses_staged_install_without_unused_triton_dependency():
    dependency = BACKEND_DEPENDENCIES["cycle_reward"]

    assert len(dependency.no_deps_requirements) == 1
    staged = dependency.no_deps_requirements[0]
    assert "cyclereward-0.1.7-py3-none-any.whl" in staged
    assert "#sha256=4a361e21454c7871" in staged
    assert "transformers==4.37.2" in dependency.requirements
    assert "timm==0.9.16" in dependency.requirements
    assert "fairscale==0.4.13" in dependency.requirements
    assert not any("triton" in requirement for requirement in dependency.requirements)


def test_legacy_image_reward_uses_compatible_torchvision_stack():
    dependency = BACKEND_DEPENDENCIES["image_reward"]

    assert "torch==2.0.1+cu118" in dependency.requirements
    assert "torchvision==0.15.2+cu118" in dependency.requirements
    assert "transformers==4.37.2" in dependency.requirements
    assert managed_pip_options(dependency)[-1].endswith("/cu118")


def test_speechbrain_uses_its_minimum_compatible_torchaudio_stack():
    for name in ("voice_identity", "voice_identity_drift"):
        dependency = BACKEND_DEPENDENCIES[name]
        assert "torch==2.7.1+cu128" in dependency.requirements
        assert "torchaudio==2.7.1+cu128" in dependency.requirements
        assert "speechbrain==1.1.2" in dependency.requirements
        assert managed_pip_options(dependency)[-1].endswith("/cu128")


def test_all_direct_insightface_consumers_use_managed_wheel_runtime():
    modules = {
        "active_speaker",
        "adaface",
        "concept_presence",
        "csim",
        "dino_face_identity",
        "entity_consistency",
        "face_cross_similarity",
        "face_emb_norm",
        "face_identity_drift",
        "facesim",
        "grafiqs",
        "identity_loss",
        "multi_subject_identity",
        "serfiq",
    }
    for module in modules:
        dependency = BACKEND_DEPENDENCIES[module]
        assert "insightface==2.1" in dependency.requirements
        assert "onnxruntime==1.31.0" in dependency.requirements
        assert dependency.python_version == "3.11"

    project = Path(__file__).parents[1] / "pyproject.toml"
    assert '"insightface>=' not in project.read_text(encoding="utf-8")


def test_mauve_audio_uses_the_imported_package_and_windows_faiss_wheel():
    dependency = BACKEND_DEPENDENCIES["mauve_audio_divergence"]
    assert dependency.requirements == ("mauve-text==0.4.0", "faiss-cpu==1.15.1")


def test_remaining_installable_backends_are_managed():
    expected = {
        "aesthetic": "aesthetic-predictor-v2-5==2024.12.18.1",
        "background_diversity": "rembg==2.0.86",
        "dedup": "ImageHash==4.3.2",
        "dreamsim": "dreamsim==0.2.1",
        "image_reward": "image-reward==1.5",
        "chamfer_sim": "open3d==0.20.0",
        "local_std_uniformity": "open3d==0.20.0",
        "mse_color": "open3d==0.20.0",
        "pc_psnr": "open3d==0.20.0",
        "voice_identity": "speechbrain==1.1.2",
        "voice_identity_drift": "speechbrain==1.1.2",
    }
    for module, requirement in expected.items():
        assert requirement in BACKEND_DEPENDENCIES[module].requirements


def test_serfiq_uses_legacy_mxnet_with_numpy_alias_compatible_runtime():
    dependency = BACKEND_DEPENDENCIES["serfiq"]
    requirements = managed_requirements(dependency)
    assert dependency.no_deps_requirements == ("mxnet==1.7.0.post2",)
    assert "numpy==1.23.5" in requirements
    assert "scikit-image==0.22.0" in requirements
    assert "insightface==2.1" in requirements
    assert "torch==2.6.0+cu124" not in requirements
    assert dependency.cpu_only


def test_vggt_alternative_uses_pinned_official_runtime():
    dependency = BACKEND_DEPENDENCIES["camera_trajectory"]
    assert dependency.config_conditions == (("pose_backend", "vggt"),)
    assert "torch==2.3.1+cu121" in dependency.requirements
    assert dependency.source_revision == "a288dd0f14786c93483e45524328726ab7b1b4ce"


def test_vendored_syncnet_consumer_has_managed_face_and_media_runtime():
    requirements = BACKEND_DEPENDENCIES["av_desync"].requirements
    assert "insightface==2.1" in requirements
    assert "onnxruntime==1.31.0" in requirements
    assert "moviepy==1.0.3" in requirements


def test_worldmodelbench_only_installs_live_vila_imports():
    requirements = BACKEND_DEPENDENCIES["worldmodelbench"].requirements
    assert requirements == (
        "hydra-core==1.3.7",
        "timm==1.0.22",
        "einops==0.8.1",
    )
    assert not any(
        "deepspeed" in item or "flash-attn" in item or "ps3" in item for item in requirements
    )
