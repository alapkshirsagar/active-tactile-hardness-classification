"""Exercise the actual model wrappers on synthetic 240x320 tactile videos.

Run from the repository root with: bash scripts/run.sh python scripts/verify_environment.py
No dataset, W&B login, or robot is required. This is an environment smoke check,
not a verification of the paper's experimental protocol.
"""

import contextlib
import ctypes
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hardness_classification"))
from models import ConvLSTM, DINOKeras, OpticalFlowNN
import adaptive_sampling
import preprocessing
import torch
import torchvision
import wandb
from scipy.signal import savgol_filter


def main():
    np_config.enable_numpy_behavior()
    tf.keras.utils.set_random_seed(42)
    torch.manual_seed(42)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow has no GPU; source scripts/env.sh first")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device("/GPU:0"):
        product = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert "GPU:0" in product.device and float(product[0, 0]) == 32.0

    cudnn = ctypes.CDLL("libcudnn.so.8")
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    runtime_version = ctypes.c_int()
    assert ctypes.CDLL("libcudart.so.11.0").cudaRuntimeGetVersion(
        ctypes.byref(runtime_version)
    ) == 0
    packages = [
        "tensorflow", "keras", "numpy", "scipy", "scikit-learn", "pandas",
        "matplotlib", "opencv-python", "torch", "torchvision", "wandb", "h5py",
    ]
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "cuda_runtime_version": runtime_version.value,
        "cudnn_runtime_version": cudnn.cudnnGetVersion(),
        "gpu": [tf.config.experimental.get_device_details(gpu) for gpu in gpus],
        "models": {},
    }
    rng = np.random.default_rng(42)
    labels = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
    with tempfile.TemporaryDirectory(prefix="tactile-env-") as temporary:
        for name, constructor, sequence_length in [
            ("ConvLSTM", ConvLSTM, 4),
            ("OFNN", OpticalFlowNN, 4),
            ("DINOv2NN", DINOKeras, 3),
        ]:
            print(f"Checking {name}: build, fit, predict, dropout, HDF5 reload", flush=True)
            videos = rng.integers(
                0, 256, size=(4, sequence_length, 240, 320, 3), dtype=np.uint8
            ).astype(np.float32)
            model = constructor()
            input_shape = videos.shape[1:]
            if name == "OFNN":
                input_shape = model.compute_optical_flow(videos[0]).shape
                assert input_shape == (500,)
            with contextlib.redirect_stdout(io.StringIO()):
                model.get_model(input_shape, 2, dropout_rate=0.2)
            model.compile_model("rmsprop", "categorical_crossentropy", ["accuracy"])
            history = model.train_model(
                videos, labels, batch_size=1, epochs=1,
                validation_samples_per_class=1, train_samples_per_class=1,
            )
            assert np.isfinite(history.history["loss"]).all()
            prediction = np.asarray(model.predict(videos[:2], batch_size=1, verbose=0))
            assert prediction.shape == (2, 2) and np.isfinite(prediction).all()
            np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-5)
            # Check the wrappers can enable dropout; sampler's missing training=True
            # for feature models is an existing code defect, not fixed here.
            stochastic = np.asarray(model.predict_raw(videos[:1], training=True))
            assert stochastic.shape == (1, 2) and np.isfinite(stochastic).all()
            path = Path(temporary) / f"{name}.h5"
            model.save_model(str(path))
            restored = tf.keras.models.load_model(path)
            original = model.model
            model.model = restored
            replay = np.asarray(model.predict(videos[:2], batch_size=1, verbose=0))
            np.testing.assert_allclose(prediction, replay, rtol=1e-5, atol=1e-6)
            model.model = original
            result["models"][name] = {
                "input_shape": list(videos.shape),
                "epochs": len(history.history["loss"]),
                "loss": float(history.history["loss"][-1]),
                "prediction_shape": list(prediction.shape),
                "weight_device": model.model.weights[0].device,
                "save_reload": "passed",
            }
            del restored, model
            tf.keras.backend.clear_session()

        smooth = savgol_filter(np.arange(10, dtype=float), 5, 2)
        plt.plot(smooth)
        plt.savefig(Path(temporary) / "plot.pdf")
        plt.close()
        with wandb.init(
            project="tactile-environment-smoke", mode="offline", dir=temporary,
            settings=wandb.Settings(start_method="thread"),
        ) as run:
            run.log({"environment_smoke": 1})

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    destination = ROOT / "environment" / "validation-output"
    destination.mkdir(exist_ok=True)
    (destination / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("PASS: GPU execution, all three models, saving/reloading, plotting, offline W&B, pip check")


if __name__ == "__main__":
    main()
