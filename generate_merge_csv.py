import os
import numpy as np
import pandas as pd

from scipy import stats as st
from sklearn.metrics import confusion_matrix


def read_csv(path):
    return pd.read_csv(path).sort_values(by=['filename'], ignore_index=True)


def main():
    ckpt_list = [
        '11-23-11-31-04_top3_submission.csv',
        '11-25-19-31-53_top3_submission.csv',
        '12-03-14-52-49_top3_submission.csv',
        '12-06-10-30-56_top3_submission.csv',
        'regent_v1.csv',
        'swin_s_1080.csv',
        'swin_b_512.csv',
        'swin_b_1080.csv',
        ]

    df_list = [read_csv(os.path.join('submission', ckpt)) for ckpt in ckpt_list]
    label_list = np.array([df['label'].to_list() for df in df_list])

    print(f'Label size: {label_list.shape}')
    labels = st.mode(label_list)[0][0]

    merge_df = pd.DataFrame({'filename': df_list[0]['filename'], 'label': labels})
    merge_df.to_csv('test.csv', index=False)
    print(f'Merge size: {labels.shape}')

if __name__ == '__main__':
    main()
