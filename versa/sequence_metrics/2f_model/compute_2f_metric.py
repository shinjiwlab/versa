#!/usr/bin/env python3

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

try:
    import matlab.engine
except ImportError:
    raise ImportError(
        "A local installation of MATLAB is required for the 2f model to run."
    )

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

TARGET_SR = 48000

def format_and_write(sig, tmp_path, orig_sr=16000):
    if not orig_sr == TARGET_SR:
        sig = librosa.resample(sig, orig_sr=orig_sr, target_sr=TARGET_SR)
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    sig = np.repeat(sig, 2, 1)
    sf.write(tmp_path, sig, TARGET_SR)


def twof_model(AvgModDiff1, ADB, clip=True):
    """
    The 2f-model is given by the following equation:
        MMSest = 56.1345 / (1 + (-0.0282 x AvgModDiff1 - 0.8628)^2) - 27.1451 x ADB + 86.3515
    """
    mms_est = (
        (56.1345 / (1 + (-0.0282 * AvgModDiff1 - 0.8628) ** 2))
        - 27.1451 * ADB
        + 86.3515
    )
    return np.clip(mms_est, 0.0, 100.0) if clip else mms_est


def calculate_2f_metric(est, ref, sr=16000):
    """
    Main routine. Given `est` and `ref`, first write both signals to disk (and resample to 48kHz)
    before computing PEAQ features / 2-f model output.
    """

    TMP_DIR = os.path.join(os.path.dirname(__file__), "2f_tmp")
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Initiate PEAQ + Matlab engine
    path_to_peaq_toolbox = (
        os.path.dirname(__file__)  # PEAQ toolbox is expected to be at the same level
    )
    m = matlab.engine.start_matlab()
    m.eval("addpath(genpath('{}'));".format(path_to_peaq_toolbox))

    # 2. Resample + stereo convert
    # (create tmp file at 48_000 as PEAQ requires to read from file directly)
    format_and_write(ref, tmp_path=os.path.join(TMP_DIR, "ref.wav"), orig_sr=sr)
    format_and_write(est, tmp_path=os.path.join(TMP_DIR, "est.wav"), orig_sr=sr)

    # 3. Compute PEAQ features necessary for 2f-model
    try:
        results = m.PQevalAudio([os.path.join(TMP_DIR, "ref.wav")], [os.path.join(TMP_DIR, "est.wav")])
        """
        MOVs is structured as follows:
            [0]:BandwidthRefB, [1]:BandwidthTestB, [2]:Total NMRB, [3]:WinModDiff1B, 
            [4]:ADBB, [5]:EHSB, [6]:AvgModDiff1B, [7]:AvgModDiff2B, [8]:RmsNoiseLoudB, 
            [9]:MFPDB, [10]:RelDistFramesB 
        """

        # 4. Get 2f-model output
        MOV = np.array(results["MOVB"][0])
        AvgModDiff1, ADB = MOV[6], MOV[4]
        mms_est = twof_model(AvgModDiff1, ADB)
    except:
        AvgModDiff1, ADB, mms_est = np.nan, np.nan, np.nan

    # Return 2-f model score along PEAQ features
    return {
        "MMS": mms_est,
        "AvgModDiff1": AvgModDiff1,
        "ADB": ADB,
    }


# debug code
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_path1 = os.path.join(script_dir, '../../../test/test_samples/test1/test.wav')
    file_path1 = os.path.normpath(file_path1)
    file_path2 = os.path.join(script_dir, '../../../test/test_samples/test2/test.wav')
    file_path2 = os.path.normpath(file_path2)
    a, sr = librosa.load(file_path1)
    b, _ = librosa.load(file_path2)
    assert np.allclose(calculate_2f_metric(a, b, sr=sr)['MMS'], 37.74, rtol=0.1)
    assert np.allclose(calculate_2f_metric(a, a, sr=sr)['MMS'], 100.0, rtol=0.1)
