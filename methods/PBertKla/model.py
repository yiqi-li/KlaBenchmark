from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


def configure_tensorflow_environment() -> None:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def prepare_proteinbert_import_path(script_dir: Path) -> Path:
    script_dir = Path(script_dir)
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return script_dir


def import_tensorflow_or_raise():
    configure_tensorflow_environment()
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        python_executable = sys.executable
        message = (
            "[PBertKla][FATAL] TensorFlow is not installed in the current Python interpreter, "
            "so ProteinBERT cannot be imported.\n"
            f"  - sys.executable = {python_executable}\n"
            "Common cause: the environment shown by package managers is not the environment running this script.\n"
            "Please either use a Python interpreter that already has TensorFlow installed, "
            "or install TensorFlow into the current interpreter."
        )
        raise ModuleNotFoundError(message) from exc
    return tf


def configure_tensorflow_memory_growth(tf) -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"[PBertKla] TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s).")
    except Exception as exc:
        print(f"[PBertKla] WARN: failed to configure TensorFlow memory growth: {exc}")


def import_proteinbert_symbols():
    from proteinbert import (
        FinetuningModelGenerator,
        OutputSpec,
        OutputType,
        load_pretrained_model,
    )
    from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

    return {
        "FinetuningModelGenerator": FinetuningModelGenerator,
        "OutputSpec": OutputSpec,
        "OutputType": OutputType,
        "load_pretrained_model": load_pretrained_model,
        "get_model_with_hidden_layers_as_outputs": get_model_with_hidden_layers_as_outputs,
    }


def ensure_pretrained_dump(pretrained_cache_dir: Path) -> Path:
    pretrained_cache_dir.mkdir(parents=True, exist_ok=True)
    dump_path = pretrained_cache_dir / "default.pkl"
    if dump_path.exists():
        return dump_path

    candidate = pretrained_cache_dir / "epoch_92400_sample_23500000.pkl"
    if not candidate.exists():
        raise FileNotFoundError(
            "[PBertKla][FATAL] Could not find ProteinBERT pretrained dump file default.pkl.\n"
            f"  - Expected path: {dump_path}\n"
            f"  - Candidate path also missing: {candidate}\n"
            "Please place epoch_92400_sample_23500000.pkl into the pretrained cache directory.\n"
            "This implementation does not download weights from the network."
        )

    try:
        if dump_path.exists() or dump_path.is_symlink():
            dump_path.unlink()
        os.symlink(str(candidate), str(dump_path))
        print(f"[PBertKla] Created symlink: {dump_path} -> {candidate}")
    except Exception as exc:
        import shutil

        shutil.copyfile(str(candidate), str(dump_path))
        print(f"[PBertKla] Symlink failed ({exc}); copied dump instead: {candidate} -> {dump_path}")
    return dump_path



def resolve_pretrained_dump_path(config) -> Path:
    cache_dir = str(getattr(config, "pretrained_cache_dir", "") or "").strip()
    if cache_dir:
        return ensure_pretrained_dump(Path(cache_dir).expanduser().resolve())

    from huggingface_hub import hf_hub_download

    repo_id = "Zakia/ProteinBERT"
    filename = "epoch_92400_sample_23500000.pkl"
    print(f"[PBertKla] Downloading ProteinBERT weights from Hugging Face: {repo_id}/{filename}")
    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    print(f"[PBertKla] Using pretrained dump: {downloaded}")
    return downloaded


def build_finetuning_components(config, script_dir: Path):
    prepare_proteinbert_import_path(script_dir)
    tf = import_tensorflow_or_raise()
    configure_tensorflow_memory_growth(tf)

    proteinbert = import_proteinbert_symbols()
    output_type = proteinbert["OutputType"](False, "binary")
    output_spec = proteinbert["OutputSpec"](output_type, unique_labels=[0, 1])

    dump_path = resolve_pretrained_dump_path(config)
    pretrained_model_generator, input_encoder = proteinbert["load_pretrained_model"](
        local_model_dump_dir=str(dump_path.parent),
        local_model_dump_file_name=dump_path.name,
        download_model_dump_if_not_exists=False,
        validate_downloading=False,
    )
    model_generator = proteinbert["FinetuningModelGenerator"](
        pretrained_model_generator,
        output_spec,
        pretraining_model_manipulation_function=proteinbert["get_model_with_hidden_layers_as_outputs"],
        dropout_rate=config.dropout,
    )
    return tf, output_spec, input_encoder, model_generator


def create_model(model_generator, seq_len: int, freeze_pretrained_layers: bool):
    return model_generator.create_model(seq_len, freeze_pretrained_layers=freeze_pretrained_layers)


def build_inference_model(model_generator, seq_len: int, weights_path: str | Path | None):
    """``seq_len`` is raw amino-acid window length (START/END added by ProteinBERT)."""
    from dataset import proteinbert_token_len

    model_generator.optimizer_weights = None
    model = create_model(
        model_generator,
        seq_len=proteinbert_token_len(seq_len),
        freeze_pretrained_layers=False,
    )
    if weights_path:
        model.load_weights(str(weights_path))
    return model


def save_encoder_artifacts(input_encoder, output_spec, input_encoder_path: Path, output_spec_path: Path) -> None:
    input_encoder_path.parent.mkdir(parents=True, exist_ok=True)
    with input_encoder_path.open("wb") as handle:
        pickle.dump(input_encoder, handle)
    with output_spec_path.open("wb") as handle:
        pickle.dump(output_spec, handle)


def load_encoder_artifacts(input_encoder_path: Path, output_spec_path: Path):
    with input_encoder_path.open("rb") as handle:
        input_encoder = pickle.load(handle)
    with output_spec_path.open("rb") as handle:
        output_spec = pickle.load(handle)
    return input_encoder, output_spec
