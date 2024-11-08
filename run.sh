stage=2

# download data
if [ $stage -eq 0 ]; then
    echo stage $stage: Prepare data

    if [ ! -d data/LibriSpeech/test-clean ]; then
        mkdir -p data
        wget http://www.openslr.org/resources/12/test-clean.tar.gz -P ./data
        (cd ./data && tar -xvzf test-clean.tar.gz)
        rm data/test-clean.tar.gz
    fi

    if [ ! -d data/LibriSpeech/test-clean/prepared ]; then
        python scripts/prepare_librispeech-test-clean.py --root_dir data/LibriSpeech/test-clean
    fi

fi

# Evaluation
pred_path=data/LibriSpeech/test-clean/prepared/ori.scp
gt_path=data/LibriSpeech/test-clean/prepared/ori.scp
if [ $stage -eq 1 ]; then
    result_path=test_result

    echo stage $stage: Evaluation
    python versa/bin/scorer.py \
        --score_config egs/speech.yaml \
        --gt ${gt_path} \
        --pred ${pred_path} \
        --output_file ${result_path}
fi

# Word error rate evaluation
# Currently use whisper_wer
if [ $stage -eq 2 ]; then
    wer_result_path=test_result_asr

    echo stage $stage: Word error rate evaluation
    python versa/bin/scorer.py \
        --score_config egs/separate_metrics/wer.yaml \
        --gt ${gt_path} \
        --pred ${pred_path} \
        --text data/LibriSpeech/test-clean/prepared/transcriptions.txt \
        --output_file ${wer_result_path}
    python scripts/get_wer.py --file_path ${wer_result_path} >> ${wer_result_path}
    tail -n 1 ${wer_result_path}
fi