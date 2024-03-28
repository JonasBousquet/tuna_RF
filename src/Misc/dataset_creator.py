import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
from rich import print

pd.options.mode.chained_assignment = None  # default='warn'


def load_data(path: str,
              sep=',',
              dec='.') -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, decimal=dec, low_memory=False)

    num_na = df.isna().any(axis=1).sum()
    num_yes = len(df)
    # df = df.dropna().reset_index()
    # df = df.drop(columns='index')
    print(f'{num_na}/{num_yes} NaNs')

    return df


def save(df, file_path):
    df.to_csv(file_path, sep=',', index=False)


def clean_data_hg(indf: pd.DataFrame) -> pd.DataFrame:
    df = indf.copy()
    df['sample_year'] = pd.to_datetime(df['sampling_date'], format='%d/%m/%Y').dt.year
    df.dropna(subset=['lon_dec'],
              inplace=True)
    df.drop(['sampling_date',
             'sample_label',
             'sample_owner',
             'scientific_name',
             'sp_category',
             'sample_identifier',
             'family', 'tissue', 'sample_position', 'sex', 'macro_matury_stage', 'c_length', 'whole_fishweight_kg',
             'water_content', 'lipid_content', 'isoNC_lab', 'd13C_pm', 'X_C', 'd15N_pm', 'X_N', 'C_N',
             'isoNC_sample_status', 'comment', 'THg_lab', 'THg_analysis_method', 'THg_sample_status'], axis=1,
            inplace=True)
    df = df[(df['c_ocean'] == 'PO') & (df['c_sp_fao'] == 'SKJ')]
    # Apply data cleaning from Medieu et al. 2022
    df.loc[df.index, 'THg_ppm_dw'] = df.apply(
        lambda row: row['THg_ww_ppb'] / 0.3 / 1000 if pd.isna(row['THg_dw_ppb']) and not pd.isna(row['THg_ww_ppb']) else
        row['THg_dw_ppb'] / 1000,
        axis=1
    )
    df['logHg'] = np.log10(df['THg_ppm_dw'])
    df = df.dropna(subset=['logHg'])
    df.drop(['EEZ', 'THg_ww_ppb'], axis=1, inplace=True)
    num_na = df.isna().any(axis=1).sum()
    num_yes = len(df)
    print(f'{num_na}/{num_yes} NaNs')
    print(f'Rip to the {num_na} rows who for some reason are missing Hg0_2x2.5 and the other things')
    df.dropna(inplace=True)
    return df


def clean_data_d13c(indf: pd.DataFrame) -> pd.DataFrame:
    df = indf.copy()
    df['sample_year'] = pd.to_datetime(df['sampling_date'], format='%d/%m/%Y').dt.year
    df.dropna(subset=['lon_dec'])
    checklist = df.columns
    filtered_df = df[~df['lat_dec'].isna() &
                     ~df['d13C_pm'].isna() &
                     ~df['sample_position'].isin(['T', 'H']) &
                     df['c_length'].isin(['FL', 'LD1', ''])]
    check_remove = ['sampling_date',
                    'sample_label',
                    'scientific_name',
                    'sp_category',
                    'sample_identifier',
                    'family', 'tissue', 'sex', 'macro_matury_stage', 'c_length', 'sample_position',
                    'whole_fishweight_kg',
                    'water_content', 'lipid_content', 'isoNC_lab', 'X_C', 'd15N_pm', 'X_N',
                    'isoNC_sample_status', 'comment', 'THg_lab', 'THg_analysis_method', 'THg_sample_status',
                    'Hg0_2x2.5', 'Hg0_6x7.5', 'drydep_2x2.5', 'drydep_6x7.5', 'wetdep_2x2.5', 'wetdep_6x7.5',
                    'totdep_2x2.5', 'totdep_6x7.5', 'THg_ww_ppb', 'THg_dw_ppb', 'EEZ']
    debug_check = [x for x in check_remove if x not in checklist]
    filtered_df.drop(check_remove, axis=1, inplace=True)

    # Applying condition to create new column 'd13C_cor'
    filtered_df['d13C_cor'] = np.where((filtered_df['C_N'] > 3.5) & ~(filtered_df['sample_owner'] == 'Madigan'),
                                       (filtered_df['d13C_pm'] + 7.489 - 7.489 * 3.097 / filtered_df['C_N']),
                                       filtered_df['d13C_pm'])
    filtered_df.drop(['sample_owner', 'C_N', 'd13C_pm'], axis=1, inplace=True)
    filtered_df.dropna(inplace=True)
    return filtered_df


def split_dataset(df: pd.DataFrame,
                  target: str):
    y = df[target]
    X = df.drop(target, axis =1)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------- MAIN ----------------------------------------------------------#
def create_datasets():
    path = '../data/DB_tuna_march2024.csv'
    savepath = '../data'

    raw_df = load_data(path=path)

    time.sleep(.5)
    hg_df = clean_data_hg(raw_df)
    print(f'Mercury dataset created n={len(hg_df)}')
    hg_path = f'{savepath}/Hg/DB_Hg_march24.csv'
    save(hg_df, hg_path)
    print(f'saved mercury data to {hg_path}')
    time.sleep(.5)

    time.sleep(.5)
    d13c_df = clean_data_d13c(raw_df)
    print(f'd13C dataset created n={len(d13c_df)}')
    d13c_path = f'{savepath}/d13C/DB_d13C_march24.csv'
    save(d13c_df, d13c_path)
    print(f'saved mercury data to {d13c_path}')


def splitter():
    basepath = '../data'
    targets = ['d13C', 'Hg']
    for i in targets:

        filepath = f'{basepath}/{i}/DB_{i}_march24.csv'
        df = pd.read_csv(filepath)

        if i == 'd13C':
            j = 'd13C_cor'
        elif i == 'Hg':
            j = 'logHg'
        X_train, X_test, y_train, y_test = split_dataset(df=df,
                                                        target=j)
        basename = f'{i}_march24.csv'
        save(X_train, f'{basepath}/{i}/X_train_{basename}')
        save(X_test, f'{basepath}/{i}/X_test_{basename}')
        save(y_train, f'{basepath}/{i}/y_train_{basename}')
        save(y_test, f'{basepath}/{i}/y_test_{basename}')


# create_datasets()
# splitter()

