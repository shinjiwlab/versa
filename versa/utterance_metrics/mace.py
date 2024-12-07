import os

import librosa
import numpy as np
from espnet2.bin.spk_inference import Speech2Embedding


def speaker_model_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = Speech2Embedding(
            model_file=model_path, train_config=model_config, device=device
        )
    else:
        if model_tag == "default":
            model_tag = "espnet/voxcelebs12_rawnet3"
        model = Speech2Embedding.from_pretrained(model_tag=model_tag, device=device)
    return model


def speaker_metric(model, pred_x, gt_x, fs):
    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    embedding_gen = model(pred_x).squeeze(0).cpu().numpy()
    embedding_gt = model(gt_x).squeeze(0).cpu().numpy()
    similarity = np.dot(embedding_gen, embedding_gt) / (
        np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
    )
    return {"spk_similarity": similarity}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    model = speaker_model_setup()
    print("metrics: {}".format(speaker_metric(model, a, b, 16000)))
