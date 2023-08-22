#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanai

"""Normalize wave data from download dir."""

import argparse
import logging
import os
import json
import h5py


import librosa
import numpy as np

from tqdm import tqdm

# from asd_tools.utils import utils#write_hdf5

def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()

def main():
    """Run preprocessing process."""
    print("Test!!!!")
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features."
    )
    parser.add_argument(
        "--download_dir",
        default=None,
        type=str,
        help="Download dir or scp file.",
    )
    parser.add_argument(
        "--dumpdir", type=str, required=True, help="directory to dump feature files."
    )
    parser.add_argument(
        "--statistic_path", type=str, default="", help="wave statistic in json file."
    )
    parser.add_argument(
        "--time_stretch_rate",
        type=float,
        default=1.0,
        help="Time stretch rate for augmentation. (default=1.0)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="If you set this flag, 'no_normalize' is set as True, wave is not normalized and saved.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    sr = 16000
    sec = 2.0
    os.makedirs(args.dumpdir, exist_ok=True)
    # get dataset
    if (len(args.statistic_path) == 0) and args.no_normalize:
        logging.info("Do not use statistics for normalizing.")
    elif not os.path.isfile(args.statistic_path):
        wave_fname_list = os.listdir(args.download_dir)
        # wave_fname_list = [fname for fname in wave_fname_list if "target" not in fname]
        tmp = np.zeros((len(wave_fname_list), 16000 * 10))
        for i, fname in enumerate(tqdm(wave_fname_list)):
            logging.info(f"fname:{fname}")
            tmp[i], _ = librosa.load(os.path.join(args.download_dir, fname), sr=sr)
        statistic = {}
        statistic["mean"] = tmp.mean()
        statistic["std"] = tmp.std()
        with open(args.statistic_path, "w") as f:
            json.dump(statistic, f)
        logging.info(f"Successfully saved statistic to {args.statistic_path}.")
    else:
        with open(args.statistic_path, "r") as f:
            statistic = json.load(f)
        logging.info(f"Successfully loaded statistic from {args.statistic_path}.")
    if not args.no_normalize:
        logging.info(
            f"Statistic mean: {statistic['mean']:.4f}, std: {statistic['std']:.4f}"
        )
    # process each data
    if args.download_dir.endswith(".scp"):
        with open(args.download_dir, "r") as f:
            wave_fname_list = [s.strip() for s in f.readlines()]
    else:
        wave_fname_list = os.listdir(args.download_dir)

    for fname in tqdm(wave_fname_list):
        if args.download_dir.endswith(".scp"):
            wav_id = fname.split("/")[-1].split(".")[0]
            path = fname
            if os.path.getsize(path) < 500000:
                logging.info(f"Size of {path} is too low!")
                continue
        else:
            wave_id_list = fname.split("_")[:6]
            if len(wave_id_list) == 3:
                wave_id_list = (
                    wave_id_list[:2]
                    + ["source", "test", "normal"]
                    + [wave_id_list[-1].split(".")[0]]
                )
            wave_id_list.insert(-1, f"{args.time_stretch_rate:.2f}")
            wav_id = "_".join(wave_id_list)
            path = os.path.join(args.download_dir, fname)
        logging.info(f"load:{path}")
        x, _ = librosa.load(path=path, sr=sr)
        if args.time_stretch_rate != 1.0:
            x = librosa.effects.time_stretch(x, args.time_stretch_rate)
        if not args.no_normalize:
            x = (x - statistic["mean"]) / statistic["std"]
        save_path = os.path.join(args.dumpdir, f"{wav_id}.h5")
        if args.download_dir.endswith(".scp"):
            x_len = len(x) - int(sr * sec)
            for i in range(5):
                start_idx = (x_len // 5) * i
                end_idx = start_idx + int(sr * sec)
                tmp_wave = x[start_idx:end_idx]
                if tmp_wave.std() == 0:
                    logging.warning(
                        f"Std of wave{i} in {save_path} is {tmp_wave.std()}."
                        "It added gaussian noise."
                    )
                    tmp_wave += np.random.randn(len(tmp_wave))
                write_hdf5(
                    save_path,
                    f"wave{i}",
                    tmp_wave.astype(np.float32),
                )
        else:
            write_hdf5(
                save_path,
                "wave",
                x.astype(np.float32),
            )


if __name__ == "__main__":
    main()
