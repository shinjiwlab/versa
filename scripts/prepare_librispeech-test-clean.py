import os
import argparse
import soundfile as sf


def normalize_transcription(transcription):
    return transcription.lower().capitalize()


def generate_files(root_dir):
    prepared_dir = os.path.join(root_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    ori_scp_path = os.path.join(prepared_dir, "ori.scp")
    transcriptions_path = os.path.join(prepared_dir, "transcriptions.txt")

    with open(ori_scp_path, "w") as ori_scp_file, open(
        transcriptions_path, "w"
    ) as transcriptions_file:
        for dir1 in os.listdir(root_dir):
            dir1_path = os.path.join(root_dir, dir1)
            if os.path.isdir(dir1_path):
                for dir2 in os.listdir(dir1_path):
                    dir2_path = os.path.join(dir1_path, dir2)
                    if os.path.isdir(dir2_path):
                        # find .txt 文件
                        for file in os.listdir(dir2_path):
                            if file.endswith(".txt"):
                                txt_file_path = os.path.join(dir2_path, file)
                                with open(txt_file_path, "r") as txt_file:
                                    # process each line
                                    for line in txt_file:
                                        parts = line.strip().split(" ", 1)
                                        if len(parts) == 2:
                                            flac_name = parts[0]
                                            transcription = parts[1]
                                            normalized_transcription = (
                                                normalize_transcription(transcription)
                                            )

                                            flac_path = os.path.join(
                                                dir2_path, f"{flac_name}.flac"
                                            )
                                            wav_path = os.path.join(
                                                dir2_path, f"{flac_name}.wav"
                                            )

                                            # transfer .flac to .wav
                                            if os.path.exists(flac_path):
                                                data, samplerate = sf.read(flac_path)
                                                sf.write(wav_path, data, samplerate)

                                                # write into ori.scp and transcriptions.txt
                                                ori_scp_file.write(
                                                    f"{flac_name} {wav_path}\n"
                                                )
                                                transcriptions_file.write(
                                                    f"{flac_name} {normalized_transcription}\n"
                                                )

    print("Files ori.scp and transcriptions.txt have been generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ori.scp and transcriptions.txt files."
    )
    parser.add_argument(
        "--root_dir", type=str, help="Root directory of the LibriSpeech dataset."
    )
    args = parser.parse_args()

    generate_files(args.root_dir)


if __name__ == "__main__":
    main()
