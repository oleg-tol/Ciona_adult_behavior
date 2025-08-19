"""
Load DLC filtered CSVs, drop likelihood columns, merge, resample, and label condition.
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import List


def process_experiment_files(folder_path: str, output_dir: str) -> pd.DataFrame:
    all_transposed: List[pd.DataFrame] = []
    file_pattern = os.path.join(folder_path, '*filtered.csv')

    for file in glob.glob(file_pattern):
        experiment_id = os.path.basename(file).split('.csv')[0]

        temp_df = pd.read_csv(file, nrows=1)
        individuals_info = temp_df.iloc[0, 1:].fillna('Unknown').values

        df = pd.read_csv(file, header=None, skiprows=1, low_memory=False)
        cols_to_keep = [i for i in range(1, df.shape[1]) if (i % 3 != 0)]
        df_filtered = df.iloc[:, cols_to_keep]

        transposed = df_filtered.T
        transposed.insert(0, 'ExperimentID', experiment_id)

        if len([x for x in individuals_info if x != 'Unknown']) == 1:
            transposed['Individuals'] = individuals_info[0]

        all_transposed.append(transposed)

    if not all_transposed:
        raise FileNotFoundError("No '*filtered.csv' files found in the folder.")

    max_len = max(d.shape[0] for d in all_transposed)
    for i, d in enumerate(all_transposed):
        if d.shape[0] < max_len:
            pad = pd.DataFrame(0, index=np.arange(max_len - d.shape[0]), columns=d.columns)
            all_transposed[i] = pd.concat([d, pad], ignore_index=True)

    merged = pd.concat(all_transposed, axis=0, ignore_index=True)
    merged.columns.values[1] = 'Individuals'
    merged.columns.values[2] = 'Bodyparts'
    merged.columns.values[3] = 'Coordinates'

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'merged_experiments.csv')
    merged.to_csv(out_path, index=False)
    return merged


def resample_df(df: pd.DataFrame, rate: int, common_cols: List[str]) -> pd.DataFrame:
    coord_cols = [c for c in df.columns if c not in common_cols]
    resampled = coord_cols[::rate]
    return df[common_cols + resampled]


def extract_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Condition'] = np.where(
        out['ExperimentID'].str.contains(r'poke', case=False, na=False), 'poke',
        np.where(out['ExperimentID'].str.contains(r'water', case=False, na=False), 'waterflow', 'control')
    )
    cols = out.columns.tolist()
    for c in ["ExperimentID", "Individuals", "Condition"]:
        if c in cols:
            cols.remove(c)
    return out[["ExperimentID", "Individuals", "Condition"] + cols]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description='Merge DLC filtered CSVs and prepare x/y data.')
    p.add_argument('folder', help='Folder with *filtered.csv files')
    p.add_argument('--outdir', default='./output_data/Ciona_adult_plotting')
    p.add_argument('--rate', type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    merged = process_experiment_files(args.folder, args.outdir)

    common = ['ExperimentID', 'Individuals', 'Bodyparts']
    df_x = merged.iloc[::2].reset_index(drop=True)
    df_y = merged.iloc[1::2].reset_index(drop=True)

    df_x_res = resample_df(df_x, args.rate, common)
    df_y_res = resample_df(df_y, args.rate, common)

    df_x_final = extract_condition(df_x_res)
    df_y_final = extract_condition(df_y_res)

    x_path = os.path.join(args.outdir, 'x_final.csv')
    y_path = os.path.join(args.outdir, 'y_final.csv')
    df_x_final.to_csv(x_path, index=False)
    df_y_final.to_csv(y_path, index=False)

    print(os.path.join(args.outdir, 'merged_experiments.csv'))
    print(x_path)
    print(y_path)


if __name__ == '__main__':
    main()