stage=1

# download data
if [ $stage -eq 0 ]; then
    echo stage $stage: Prepare data

    # librispeech
    if [ ! -d data//LibriSpeech/test-clean ]; then
        mkdir -p data/
        wget http://www.openslr.org/resources/12/test-clean.tar.gz -P data/
        (cd data/ && tar -xvzf test-clean.tar.gz)
        rm data/test-clean.tar.gz
    fi

    if [ ! -d data//LibriSpeech/test-clean/prepared ]; then
        python scripts/prepare_librispeech-test-clean.py --root_dir data//LibriSpeech/test-clean
    fi

    # musdb
    if [ ! -d data//musdb/test ]; then
        wget https://zenodo.org/records/3338373/files/musdb18hq.zip -P data/
        (cd data/ && unzip musdb18hq.zip -d ../musdb)
        rm data/musdb18hq.zip
    fi

    if [ ! -d data//musdb/prepared ]; then
        python scripts/prepare_musdb.py --main_directory data//musdb/ --output_dir data//musdb/prepared --chunk_length 5.0
    fi

    # audioset
    if [ ! -d data//audioset ]; then
        python scripts/prepare_audioset-test.py --output_dir data//audioset/
    fi

fi

# Evaluation
pred_path=data/LibriSpeech/test-clean/prepared/ori.scp
gt_path=data/LibriSpeech/test-clean/prepared/ori.scp
tag=musdb_encodec_24k_12bps
eval_sr=24000
if [ $stage -eq 1 ]; then
    result_path="test_result_${tag}"

    echo stage $stage: Evaluation
    if test -f ${result_path}; then
        echo ${result_path} exists
    else
        python versa/bin/scorer.py \
            --score_config egs/general.yaml \
            --use_gpu True \
            --gt ${gt_path} \
            --pred ${pred_path} \
            --output_file ${result_path} \
            --eval_sr ${eval_sr} \  # change in versa necessary!
    fi

    python scripts/average_result.py --file_path ${result_path} >> ${result_path}
    tail -n 20 ${result_path}
fi

# Word error rate evaluation
# Currently use whisper_wer
if [ $stage -eq 2 ]; then
    wer_result_path="test_result_asr"

    echo stage $stage: Word error rate evaluation

    if test -f ${wer_result_path}; then
        echo ${wer_result_path} exists
    else
        python versa/bin/scorer.py \
            --score_config egs/separate_metrics/wer.yaml \
            --use_gpu True \
            --gt ${gt_path} \
            --pred ${pred_path} \
            --text data/LibriSpeech/test-clean/prepared/transcriptions.txt \
            --output_file ${wer_result_path}
    fi

    python scripts/get_wer.py --file_path ${wer_result_path} >> ${wer_result_path}
    tail -n 1 ${wer_result_path}
fi