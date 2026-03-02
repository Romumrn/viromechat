#!/usr/bin/python3
import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TAXO_DB_PATH, HOST_DB_PATH

# ==================== DATA LOADING ==================== #

@st.cache_data(show_spinner=False)
def load_dataframe(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ==================== HELPERS ==================== #

def metric_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, str(value))


def safe_nunique(df, col):
    """Return nunique as formatted string, or 'N/A' if column missing."""
    if col in df.columns:
        return f"{df[col].nunique():,}"
    return "N/A"


def filterable_table(df: pd.DataFrame, key_prefix: str):
    """Display a searchable, column-filtered dataframe."""
    search = st.text_input("🔍 Search (any column)", key=f"{key_prefix}_search")

    col_filter, col_vals = st.columns([1, 2])
    with col_filter:
        filter_col = st.selectbox(
            "Filter by column",
            options=["— none —"] + list(df.columns),
            key=f"{key_prefix}_col"
        )
    with col_vals:
        if filter_col != "— none —":
            unique_vals = sorted(df[filter_col].dropna().astype(str).unique())
            selected_vals = st.multiselect("Values", options=unique_vals, key=f"{key_prefix}_vals")
        else:
            selected_vals = []

    filtered = df.copy()

    if search:
        mask = filtered.apply(
            lambda c: c.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        filtered = filtered[mask]

    if filter_col != "— none —" and selected_vals:
        filtered = filtered[filtered[filter_col].astype(str).isin(selected_vals)]

    total = len(filtered)
    PAGE_SIZE = 200

    st.caption(f"Showing **{min(PAGE_SIZE, total):,}** of **{total:,}** / {len(df):,} rows  _(display limited to {PAGE_SIZE} rows — use download for full data)_")
    st.dataframe(filtered.head(PAGE_SIZE), use_container_width=True, height=500)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download all {total:,} rows as CSV",
        data=csv,
        file_name=f"{key_prefix}_filtered.csv",
        mime="text/csv",
        key=f"{key_prefix}_download"
    )


# ==================== MAIN ==================== #

def main():
    st.set_page_config(page_title="Database Viewer 🦠", page_icon="🗄️", layout="wide")
    st.title(" Database Viewer")
    st.caption("Explore the raw datasets.")

    df_taxo = load_dataframe(TAXO_DB_PATH)
    df_host  = load_dataframe(HOST_DB_PATH)

    if df_taxo is None or df_host is None:
        st.error("One or more dataset files not found. Check your `data/` directory.")
        st.stop()

    # Global statistics
    st.header("Global Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Taxonomy (df_taxo)")
        metric_row([
            ("Total entries", f"{len(df_taxo):,}"),
            ("Families",      safe_nunique(df_taxo, "FAMILY_NAME")),
            ("Genera",        safe_nunique(df_taxo, "GENUS_NAME")),
            ("Species",       safe_nunique(df_taxo, "SPECIES_NAME")),
        ])

    with col2:
        st.subheader("Virus-Host (df_host)")
        
        metric_row([
            ("Total entries", f"{len(df_taxo):,}"),
            ("Viruses",   safe_nunique(df_host, "VIRAL_TAX_ID") )  ,
            ("Hosts",   safe_nunique(df_host, "TAX_ID") )  ,
            ("Location", df_host["LOCALISATION_RESOLUTION"].count()),
        ])


    st.divider()

    # Tabs
    tab_taxo, tab_host = st.tabs(["Taxonomy", "Virus-Host"])

    with tab_taxo:
        st.subheader("Taxonomy Database")
        st.caption(f"Source: `{TAXO_DB_PATH}` — {len(df_taxo):,} rows × {len(df_taxo.columns)} columns")
        filterable_table(df_taxo, key_prefix="taxo")

    with tab_host:
        st.subheader("Virus-Host Database")
        st.caption(f"Source: `{HOST_DB_PATH}` — {len(df_host):,} rows × {len(df_host.columns)} columns")
        filterable_table(df_host, key_prefix="host")


if __name__ == "__main__":
    main()