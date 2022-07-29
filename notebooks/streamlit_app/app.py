import streamlit as st
import pandas as pd
import numpy as np


def show_notebook(g):
    st.write('Kendall_tau is ', g['kendall_tau'].iloc[0])
    st.dataframe(g[['cell_id', 'cell_type', 'source']])

    for (ct, src) in g[['cell_type', 'source']].to_numpy():
        if ct == 'markdown':
            st.markdown(src)
        else:
            st.code(src)


def main():
    df_kt = pd.read_csv('../../data/kt_results.csv')
    df = pd.read_feather('../../data/val_df')
    df = pd.merge(df, df_kt, left_on='id', right_on='index', how='inner')
    grouped = df.groupby('id')

    kt_threshold = st.slider(
         'Select maximum not bad kendall_tau',
         -1.0, 1.0, 0.5, 0.1
    )

    notebook_ids = df[df.kendall_tau <= kt_threshold]['id'].unique()

    st.title('Bad notebooks')

    option = st.selectbox(
        'Notebook id',
        notebook_ids[:50])

    st.write('You selected:', option)
    show_notebook(grouped.get_group(option))


if __name__=='__main__':
    main()
