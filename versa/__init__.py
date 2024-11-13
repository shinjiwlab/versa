import logging

from versa.sequence_metrics.mcd_f0 import mcd_f0
from versa.sequence_metrics.signal_metric import signal_metric

try:
    from versa.utterance_metrics.discrete_speech import (
        discrete_speech_metric, discrete_speech_setup)
except ImportError:
    logging.warning(
        "Please pip install git+https://github.com/ftshijt/DiscreteSpeechMetrics.git and retry"
    )

from versa.utterance_metrics.pseudo_mos import (pseudo_mos_metric,
                                                pseudo_mos_setup)

try:
    from versa.utterance_metrics.pesq_score import pesq_metric
except ImportError:
    logging.warning("Please install pesq with `pip install pesq` and retry")

try:
    from versa.utterance_metrics.stoi import stoi_metric
except ImportError:
    logging.warning("Please install pystoi with `pip install pystoi` and retry")

try:
    from versa.utterance_metrics.speaker import (speaker_metric,
                                                 speaker_model_setup)
except ImportError:
    logging.warning("Please install espnet with `pip install espnet` and retry")

try:
    from versa.utterance_metrics.visqol_score import (visqol_metric,
                                                      visqol_setup)
except ImportError:
    logging.warning(
        "Please install visqol follow https://github.com/google/visqol and retry"
    )

from versa.corpus_metrics.espnet_wer import (espnet_levenshtein_metric,
                                             espnet_wer_setup)
from versa.corpus_metrics.owsm_wer import (owsm_levenshtein_metric,
                                           owsm_wer_setup)
from versa.corpus_metrics.whisper_wer import (whisper_levenshtein_metric,
                                              whisper_wer_setup)
from versa.utterance_metrics.sheet_ssqa import sheet_ssqa, sheet_ssqa_setup
from versa.utterance_metrics.squim import squim_metric, squim_metric_no_ref
from versa.utterance_metrics.scoreq import scoreq_nr, scoreq_nr_setup, scoreq_ref, scoreq_ref_setup