from setuptools import setup, find_packages

setup(
    name='speech-evaluation-toolkit',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.3',
        'pysptk>=0.1.19',
        'pyworld>=0.3.0',
        'fastdtw>=0.3.4',
        'pystoi>=0.0.1',
        'pesq>=0.0.4',
        'fast_bss_eval>=0.1.4',
        'ci_sdr>=0.0.2',
        'mir_eval>=0.7',
        'speechmos>0.0.1.1',
        'scipy>=1.7.1',
        'pypesq>=1.2.4',
        'librosa>=0.8.1',
        'transformers>=4.36.2',
        'torch>=1.10.1',       # https://github.com/huggingface/transformers/issues/26796
        'torchaudio>=0.10.1',  # In torch >= 2.0.0, warnings for checkpoint mismatch are raised.
        'joblib>=1.0.1',
        'nltk>=3.6.5',
        'Levenshtein>=0.23.0',
        'jellyfish>=1.0.3',
    ],
    author='Jiatong Shi',
    author_email='ftshijt@gmail.com',
    description='A package for computing speech evaluation metrics.',
    url='https://github.com/ftshijt/speech_evaluation',
    keywords='speech metrics',
)