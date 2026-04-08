# -*- coding: utf-8 -*-
"""
WFU MSBA Practicum - Regulatory Anomaly Detection Dashboard
==========================================================
Layperson-friendly Streamlit dashboard for exploring per-bank anomaly
detection results across FFIEC Call Reports and FR Y-9C filings.

Author: Wake Forest MSBA Practicum Team 4
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import threading
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DATA_DIR = _PROJECT_ROOT / "output_data" / "ffiec" / "nn_anomalies"
LEVELS_DIR = _PROJECT_ROOT / "output_data" / "ffiec" / "nn_levels_anomalies"
QOQ_RAW_DIR = _PROJECT_ROOT / "ffiec_call_reports" / "04_qoq" / "per_bank_qoq"
FEATURES_RAW_DIR = _PROJECT_ROOT / "input_data" / "ffiec" / "per_bank_features"
ISO_DIR = _PROJECT_ROOT / "output_data" / "ffiec" / "iso_output"
ISO_LEVELS_DIR = _PROJECT_ROOT / "output_data" / "ffiec" / "iso_output"
FRY9C_QOQ_DIR = _PROJECT_ROOT / "output_data" / "fry9c" / "nn_anomalies"
FRY9C_LEVELS_DIR = _PROJECT_ROOT / "output_data" / "fry9c" / "nn_levels_anomalies"
FRY9C_ISO_DIR = _PROJECT_ROOT / "output_data" / "fry9c" / "iso_output"
FORECAST_DIR = _PROJECT_ROOT / "output_data" / "ffiec" / "forecasts"

FORECAST_CUTOFF = pd.Timestamp("2025-09-30")

FORECAST_BANK_FILES = {
    "Bank of America":        "BofaForecasted.csv",
    "Citibank":               "CitiForecasted.csv",
    "JPMorgan Chase Bank":    "JPMorganForecasted.csv",
    "Goldman Sachs Bank USA": "ffiec_goldman_sachs_arima112_forecast.csv",
    "Wells Fargo Bank":       "ffiec_wells_fargo_arima112_forecast.csv",
    "Morgan Stanley Bank":    "morgan_stanley_forecast.csv",
}

BANK_DISPLAY_NAMES = {
    'bank_of_america': 'Bank of America',
    'jpmorgan_chase_bank': 'JPMorgan Chase Bank',
    'citibank': 'Citibank',
    'wells_fargo_bank': 'Wells Fargo Bank',
    'goldman_sachs_bank_usa': 'Goldman Sachs Bank USA',
    'morgan_stanley_bank': 'Morgan Stanley Bank',
}

FRY9C_DISPLAY_NAMES = {
    'bank_of_america_corporation':  'Bank of America Corporation',
    'jpmorgan_chase_and_co':        'JPMorgan Chase & Co.',
    'citigroup_inc':                'Citigroup Inc.',
    'wells_fargo_and_company':      'Wells Fargo & Company',
    'goldman_sachs_group_inc':      'Goldman Sachs Group Inc.',
    'morgan_stanley':               'Morgan Stanley',
}

# Map bank_name values in ISO output CSVs -> dashboard display names
_FRY9C_ISO_NAME_MAP = {
    "JPMorgan Chase and Co":        "JPMorgan Chase & Co.",
    "Citigroup Inc":                "Citigroup Inc.",
    "Bank of America Corporation":  "Bank of America Corporation",
    "Wells Fargo and Company":      "Wells Fargo & Company",
    "Morgan Stanley":               "Morgan Stanley",
    "Goldman Sachs Group Inc":      "Goldman Sachs Group Inc.",
}

ANOMALY_RED = '#E63946'
NORMAL_BLUE = '#457B9D'
NN_GREEN = '#2A9D8F'
LOF_GOLD = '#E9C46A'
ISO_LVL_PURPLE = '#9B59B6'

# Columns to strip when extracting raw features from anomaly CSVs
_SCORE_COLS = {
    'nn_score', 'lof_score', 'ensemble_score', 'nn_anomaly',
    'lof_anomaly', 'is_anomaly', 'reconstruction_error',
    'anomaly_score', 'is_anomaly_adaptive', 'quarter', 'quarter_date',
}

UNIT_LABELS = {'F': '$ thousands', 'P': '%', 'R': 'Ratio', 'S': 'Count', 'D': 'Date'}


# =============================================================================
# HELPERS
# =============================================================================

def _score_col(df):
    for col in ('ensemble_score', 'anomaly_score', 'reconstruction_error'):
        if col in df.columns:
            return col
    return None


def _ensure_score_cols(df):
    if 'ensemble_score' in df.columns and 'anomaly_score' not in df.columns:
        df['anomaly_score'] = df['ensemble_score']
    elif 'anomaly_score' in df.columns and 'ensemble_score' not in df.columns:
        df['ensemble_score'] = df['anomaly_score']
    elif 'reconstruction_error' in df.columns and 'anomaly_score' not in df.columns:
        df['anomaly_score'] = df['reconstruction_error']
        df['ensemble_score'] = df['reconstruction_error']
    return df


def get_unit_label(code, type_dict):
    clean = code.replace('_qoq', '')
    return UNIT_LABELS.get(type_dict.get(clean, 'F'), '')


def _truncate(text, max_len=55):
    return text if len(text) <= max_len else text[:max_len - 1].rstrip() + "..."


def translate_code(code, mdrm_dict, truncate=False):
    clean = code.replace('_qoq', '')
    name = mdrm_dict.get(clean, '')
    if name:
        display = _truncate(name) if truncate else name
        return f"{display} ({clean})"
    return clean


def chart_label(code, mdrm_dict):
    clean = code.replace('_qoq', '')
    name = mdrm_dict.get(clean, '')
    return f"{_truncate(name, 45)} ({clean})" if name else clean


def short_name(code, mdrm_dict):
    clean = code.replace('_qoq', '')
    return mdrm_dict.get(clean, '') or clean


def parse_year(quarter_str):
    try:
        return int(str(quarter_str).split('/')[-1])
    except (ValueError, IndexError):
        return None


def preferred_entity_name(is_fry):
    return "Bank of America Corporation" if is_fry else "Bank of America"


def latest_quarter_dates_for_bank(bank_name, data_dicts):
    dates = []
    for data_dict in data_dicts:
        df = data_dict.get(bank_name) if isinstance(data_dict, dict) else None
        if df is None or 'quarter_date' not in df.columns:
            continue
        dates.extend(df['quarter_date'].dropna().tolist())

    if not dates:
        return []

    return sorted(pd.Series(dates).dropna().drop_duplicates().tolist())


def default_year_window_for_bank(bank_name, data_dicts, min_year, max_year, recent_quarters=8):
    bank_dates = latest_quarter_dates_for_bank(bank_name, data_dicts)
    if len(bank_dates) < recent_quarters:
        return (min_year, max_year)

    recent_dates = bank_dates[-recent_quarters:]
    return (
        max(min_year, recent_dates[0].year),
        min(max_year, recent_dates[-1].year),
    )


def ordered_quarters_for_detail(bank_conv):
    if bank_conv.empty:
        return []

    all_qs = bank_conv.sort_values('quarter_date', ascending=False)['Quarter'].tolist()
    flagged_qs = bank_conv[bank_conv['Models Flagged'] >= 1].sort_values(
        'quarter_date', ascending=False
    )['Quarter'].tolist()

    ordered = []
    for quarter in all_qs[:1] + flagged_qs + all_qs:
        if quarter not in ordered:
            ordered.append(quarter)
    return ordered


def format_flagged_models(flags):
    return ", ".join(flags) if flags else "None"


def source_warning_text(label, path):
    return f"{label} source data is unavailable. Expected files in `{path}`."


def build_driver_table(bank_raw, quarter, mdrm, top_n=15):
    if bank_raw is None or bank_raw.empty or 'quarter' not in bank_raw.columns:
        return pd.DataFrame()
    if quarter not in bank_raw['quarter'].values:
        return pd.DataFrame()

    feat_skip = {'quarter', 'quarter_date'}
    feat_cols = [c for c in bank_raw.columns if c not in feat_skip and not c.startswith('_')]
    if not feat_cols:
        return pd.DataFrame()

    feat_data = bank_raw[feat_cols].apply(pd.to_numeric, errors='coerce')
    bank_means = feat_data.mean()
    bank_stds = feat_data.std().replace(0, np.nan)
    row_vals = feat_data[bank_raw['quarter'] == quarter].iloc[0]
    z_scores = ((row_vals - bank_means) / bank_stds).dropna()
    top_features = z_scores.abs().nlargest(top_n)
    if top_features.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        'raw_code': top_features.index,
        'Line Item': [translate_code(f, mdrm) for f in top_features.index],
        'Chart Label': [chart_label(f, mdrm) for f in top_features.index],
        'Value': [row_vals[f] for f in top_features.index],
        'Average': [bank_means[f] for f in top_features.index],
        'How Unusual': [z_scores[f] for f in top_features.index],
    })


def plotly_theme_layout():
    return dict(
        template="plotly_dark" if _is_dark_theme() else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', size=12),
        margin=dict(l=20, r=20, t=30, b=40),
    )


def _is_dark_theme():
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True


def _extract_features_from_anomaly_df(bank_dict):
    """Extract raw feature data from anomaly CSVs (strip score/flag cols)."""
    result = {}
    for bname, bdf in bank_dict.items():
        keep = ['quarter', 'quarter_date'] + [c for c in bdf.columns if c not in _SCORE_COLS]
        result[bname] = bdf[keep].copy()
    return result


def compute_cross_bank_zscores(raw_dict, conv_df, threshold=2.0):
    """Compute z-scores for flagged quarters across all banks.

    For each bank, computes feature z-scores against that bank's own
    historical mean/std.  Returns only deviations exceeding *threshold*.

    Parameters
    ----------
    raw_dict : dict[str, DataFrame]
        bank_name -> DataFrame with feature columns + quarter/quarter_date.
    conv_df : DataFrame
        Convergence table with Bank, Quarter, Models Flagged columns.
    threshold : float
        Minimum |z| to include in results.

    Returns
    -------
    DataFrame with columns: bank, quarter, quarter_date, feature_code,
        z_score, value, historical_mean.
    """
    if conv_df.empty:
        return pd.DataFrame(columns=[
            'bank', 'quarter', 'quarter_date', 'feature_code',
            'z_score', 'value', 'historical_mean',
        ])

    flagged = conv_df[conv_df['Models Flagged'] >= 1][['Bank', 'Quarter']].copy()
    flagged_set = set(zip(flagged['Bank'], flagged['Quarter']))

    feat_skip = {'quarter', 'quarter_date'}
    rows = []
    for bname, bdf in raw_dict.items():
        feat_cols = [c for c in bdf.columns
                     if c not in feat_skip and not c.startswith('_')]
        if not feat_cols:
            continue
        feat_data = bdf[feat_cols].apply(pd.to_numeric, errors='coerce')
        bank_means = feat_data.mean()
        bank_stds = feat_data.std().replace(0, np.nan)

        for _, r in bdf.iterrows():
            q = r['quarter']
            if (bname, q) not in flagged_set:
                continue
            vals = pd.to_numeric(bdf.loc[bdf['quarter'] == q, feat_cols].iloc[0],
                                 errors='coerce')
            z = ((vals - bank_means) / bank_stds).dropna()
            notable = z[z.abs() > threshold]
            for code, zscore in notable.items():
                rows.append({
                    'bank': bname,
                    'quarter': q,
                    'quarter_date': r.get('quarter_date', pd.NaT),
                    'feature_code': code,
                    'z_score': zscore,
                    'value': vals[code],
                    'historical_mean': bank_means[code],
                })

    if not rows:
        return pd.DataFrame(columns=[
            'bank', 'quarter', 'quarter_date', 'feature_code',
            'z_score', 'value', 'historical_mean',
        ])
    return pd.DataFrame(rows)


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_lookup_csv(path):
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path, dtype=str)
    name_dict = dict(zip(df['code'], df['description']))
    type_dict = {}
    if 'item_type' in df.columns:
        type_dict = dict(zip(df['code'], df['item_type'].fillna('F')))
    return name_dict, type_dict


@st.cache_data
def load_mdrm_lookup():
    ffiec_path = _PROJECT_ROOT / "shared" / "mdrm_lookup.csv"
    if not ffiec_path.exists():
        ffiec_path = DATA_DIR / "mdrm_lookup.csv"
    name_dict, type_dict = _load_lookup_csv(ffiec_path)
    fry_path = _PROJECT_ROOT / "shared" / "fry9c_lookup.csv"
    fry_names, fry_types = _load_lookup_csv(fry_path)
    name_dict.update(fry_names)
    type_dict.update(fry_types)
    return name_dict, type_dict


def _load_bank_csvs(directory, glob_pattern):
    banks = {}
    if not Path(directory).exists():
        return banks
    for filepath in sorted(Path(directory).glob(glob_pattern)):
        if 'flagged' in filepath.stem:
            continue
        slug = filepath.stem
        for prefix in ['ffiec_']:
            slug = slug.replace(prefix, '')
        for suffix in ['_nn_anomalies', '_nn_levels_anomalies']:
            slug = slug.replace(suffix, '')
        display_name = BANK_DISPLAY_NAMES.get(slug, slug.replace('_', ' ').title())
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception:
            try:
                df = pd.read_excel(filepath, engine='xlrd')
            except Exception:
                continue
        if 'feature' in df.columns:
            feature_names = df['feature'].tolist()
            df_t = df.drop(columns=['feature']).T
            df_t.columns = feature_names
            df_t.index.name = 'quarter'
            df_t = df_t.reset_index()
            df_t.rename(columns={'index': 'quarter'}, inplace=True)
            df = df_t
        for col in ['nn_score', 'lof_score', 'ensemble_score', 'reconstruction_error', 'anomaly_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['nn_anomaly', 'lof_anomaly', 'is_anomaly']:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        df = _ensure_score_cols(df)
        if 'quarter' in df.columns:
            df['quarter_date'] = pd.to_datetime(df['quarter'], format='%m/%d/%Y', errors='coerce')
            df = df.sort_values('quarter_date').reset_index(drop=True)
        banks[display_name] = df
    return banks


@st.cache_data
def load_all_banks():
    return _load_bank_csvs(DATA_DIR, "ffiec_*_nn_anomalies.*")


@st.cache_data
def load_levels_banks():
    return _load_bank_csvs(LEVELS_DIR, "ffiec_*_nn_levels_anomalies.csv")


def _load_fry9c_csvs(directory, glob_pattern, suffix_strip):
    banks = {}
    if not Path(directory).exists():
        return banks
    for filepath in sorted(Path(directory).glob(glob_pattern)):
        if 'flagged' in filepath.stem:
            continue
        slug = filepath.stem.replace('fry9c_', '').replace(suffix_strip, '')
        display_name = FRY9C_DISPLAY_NAMES.get(slug, slug.replace('_', ' ').title())
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception:
            continue
        for col in ['nn_score', 'lof_score', 'ensemble_score', 'reconstruction_error', 'anomaly_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['nn_anomaly', 'lof_anomaly', 'is_anomaly']:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        df = _ensure_score_cols(df)
        if 'quarter' in df.columns:
            df['quarter_date'] = pd.to_datetime(df['quarter'], format='%m/%d/%Y', errors='coerce')
            df = df.sort_values('quarter_date').reset_index(drop=True)
        banks[display_name] = df
    return banks


@st.cache_data
def load_fry9c_qoq():
    return _load_fry9c_csvs(FRY9C_QOQ_DIR, "fry9c_*_nn_anomalies.csv", "_nn_anomalies")


@st.cache_data
def load_fry9c_levels():
    return _load_fry9c_csvs(FRY9C_LEVELS_DIR, "fry9c_*_nn_levels_anomalies.csv", "_nn_levels_anomalies")


def _load_iso(path, name_map=None):
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    for col in ('anom_score', 'systemic_score', 'adjusted_score'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'is_anomaly' in df.columns:
        df['is_anomaly'] = df['is_anomaly'].astype(bool)
    if name_map and 'bank_name' in df.columns:
        df['bank_name'] = df['bank_name'].map(lambda x: name_map.get(x, x))
    if 'quarter' in df.columns:
        df['quarter_date'] = pd.to_datetime(df['quarter'], format='%m/%d/%Y', errors='coerce')
        df = df.sort_values(['bank_name', 'quarter_date']).reset_index(drop=True)
    df['ensemble_score'] = df['anom_score']
    df['anomaly_score'] = df['adjusted_score']
    return {name: g.reset_index(drop=True) for name, g in df.groupby('bank_name')}


@st.cache_data
def load_iso_banks():
    """FFIEC Isolation Forest - absolute levels."""
    return _load_iso(ISO_DIR / "ffiec_iso_absolute_all_results.csv")


@st.cache_data
def load_iso_qoq_banks():
    """FFIEC Isolation Forest - QoQ changes."""
    return _load_iso(ISO_DIR / "ffiec_iso_qoq_all_results.csv")


@st.cache_data
def load_iso_levels_banks():
    return _load_iso(ISO_LEVELS_DIR / "ffiec_iso_absolute_all_results.csv")


@st.cache_data
def load_fry9c_iso_levels():
    """FR Y-9C Isolation Forest - absolute levels."""
    return _load_iso(FRY9C_ISO_DIR / "fry9c_iso_absolute_all_results.csv", _FRY9C_ISO_NAME_MAP)


@st.cache_data
def load_fry9c_iso_qoq():
    """FR Y-9C Isolation Forest - QoQ changes."""
    return _load_iso(FRY9C_ISO_DIR / "fry9c_iso_qoq_all_results.csv", _FRY9C_ISO_NAME_MAP)


@st.cache_data
def load_iso_levels_findings():
    fpath = ISO_LEVELS_DIR / "ffiec_iso_absolute_findings.csv"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    for col in ('anom_score', 'systemic_score', 'adjusted_score'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data
def load_iso_levels_stability():
    fpath = ISO_LEVELS_DIR / "ffiec_iso_absolute_stability.csv"
    return pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()


@st.cache_data
def load_iso_levels_precision():
    fpath = ISO_LEVELS_DIR / "ffiec_iso_absolute_precision_audit.csv"
    return pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()


@st.cache_data
def load_raw_data(data_dir, suffix):
    banks = {}
    if not Path(data_dir).exists():
        return banks
    for filepath in sorted(Path(data_dir).glob(f"ffiec_*_{suffix}.csv")):
        slug = filepath.stem.replace('ffiec_', '').replace(f'_{suffix}', '')
        display_name = BANK_DISPLAY_NAMES.get(slug, slug.replace('_', ' ').title())
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception:
            continue
        if 'feature' in df.columns:
            feature_names = df['feature'].tolist()
            df_t = df.drop(columns=['feature']).T
            df_t.columns = feature_names
            df_t.index.name = 'quarter'
            df_t = df_t.reset_index()
            df_t.rename(columns={'index': 'quarter'}, inplace=True)
            df = df_t
        if 'quarter' in df.columns:
            df['quarter_date'] = pd.to_datetime(df['quarter'], format='%m/%d/%Y', errors='coerce')
            df = df.sort_values('quarter_date').reset_index(drop=True)
        for col in df.columns:
            if col not in ('quarter', 'quarter_date'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        banks[display_name] = df
    return banks


@st.cache_data
def load_forecasts(bank_name):
    fname = FORECAST_BANK_FILES.get(bank_name)
    if not fname:
        return None
    fpath = FORECAST_DIR / fname
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath, index_col=0)
    except Exception:
        return None
    df = df.reset_index().rename(columns={"index": "feature"})
    df_long = df.melt(id_vars="feature", var_name="quarter", value_name="value")
    df_long["quarter_date"] = pd.to_datetime(df_long["quarter"], errors="coerce")
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long[df_long["quarter_date"] > FORECAST_CUTOFF].copy()
    return df_long.sort_values(["feature", "quarter_date"]).reset_index(drop=True)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="WFU MSBA Practicum - Regulatory Anomaly Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .main .block-container {
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# REFRESH DATA
# =============================================================================

def _refresh_data_ui():
    """Render the Refresh Data button and handle pipeline execution."""

    # --- Confirmation state ---
    if "confirm_refresh" not in st.session_state:
        st.session_state.confirm_refresh = False

    if not st.session_state.confirm_refresh:
        if st.button("Refresh Data", type="primary", use_container_width=True,
                     help="Download latest quarters, re-run models, update results"):
            st.session_state.confirm_refresh = True
            st.rerun()
        return

    # --- Confirmation dialog ---
    st.warning(
        "**Are you sure?** This will scrape the FFIEC & Chicago Fed APIs and "
        "rerun all models. The dashboard will be unresponsive for **3-7 minutes**.",
    )
    col1, col2 = st.columns(2)
    with col1:
        confirmed = st.button("Yes, Refresh", type="primary", use_container_width=True)
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.confirm_refresh = False
            st.rerun()

    if not confirmed:
        return

    # --- Run pipeline ---
    st.session_state.confirm_refresh = False

    project_root = _PROJECT_ROOT
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from refresh_pipeline import run as run_pipeline
    except ImportError as e:
        st.error(f"Could not import refresh pipeline: {e}")
        return

    with st.status("Refreshing data...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        log_area = st.empty()

        steps = [
            "Scraping FR Y-9C", "Scraping FFIEC",
            "Splitting FR Y-9C", "Splitting FFIEC",
            "FR Y-9C Feature Selection", "FFIEC Feature Selection",
            "FR Y-9C QoQ", "FFIEC QoQ",
            "FR Y-9C Neural Network (QoQ)", "FR Y-9C Neural Network (Levels)",
            "FR Y-9C Isolation Forest", "FR Y-9C Isolation Forest (QoQ)",
            "FFIEC Neural Network (QoQ)", "FFIEC Neural Network (Levels)",
            "FFIEC Isolation Forest", "FFIEC Isolation Forest (QoQ)",
        ]
        total_steps = len(steps)

        def progress_callback(step_name, message, step_pct):
            # Map step name to overall progress
            try:
                step_idx = steps.index(step_name)
                overall = (step_idx + step_pct) / total_steps
            except ValueError:
                overall = step_pct
            progress_bar.progress(min(overall, 1.0))
            log_area.text(f"{step_name}: {message}")

        result = run_pipeline(progress_callback=progress_callback)

        progress_bar.progress(1.0)

        if result["success"]:
            status.update(label="Refresh complete!", state="complete")
            st.success(result["summary"])
        else:
            status.update(label="Refresh completed with errors", state="error")
            st.warning(result["summary"])
            for err in result["errors"]:
                st.error(err)

    # Clear cached data and rerun to show updated results
    st.cache_data.clear()
    st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title("WFU MSBA Practicum - Regulatory Anomaly Detection Dashboard")
    st.caption("WFU MSBA Anomaly Detection Model")

    # =========================================================================
    # LOAD ALL DATA
    # =========================================================================
    ffiec_qoq    = load_all_banks()
    ffiec_levels = load_levels_banks()
    ffiec_iso_lvl = load_iso_banks()
    ffiec_iso_qoq = load_iso_qoq_banks()
    ffiec_raw    = load_raw_data(FEATURES_RAW_DIR, "features")
    ffiec_raw_qoq = load_raw_data(QOQ_RAW_DIR, "qoq")
    mdrm, mdrm_types = load_mdrm_lookup()

    iso_lvl_findings  = load_iso_levels_findings()
    iso_lvl_stability = load_iso_levels_stability()
    iso_lvl_precision = load_iso_levels_precision()

    fry_qoq    = load_fry9c_qoq()
    fry_levels = load_fry9c_levels()
    fry_iso_lvl = load_fry9c_iso_levels()
    fry_iso_qoq = load_fry9c_iso_qoq()

    has_ffiec = len(ffiec_qoq) > 0 or len(ffiec_levels) > 0 or len(ffiec_iso_lvl) > 0 or len(ffiec_iso_qoq) > 0
    has_fry   = len(fry_qoq) > 0 or len(fry_levels) > 0 or len(fry_iso_lvl) > 0 or len(fry_iso_qoq) > 0

    if not has_ffiec and not has_fry:
        st.error("No model output found.")
        return

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("Settings")

        # --- Refresh Data Button ---
        _refresh_data_ui()

        st.divider()

        # --- Report type toggle ---
        report_options = []
        if has_ffiec:
            report_options.append("FFIEC Call Reports")
        if has_fry:
            report_options.append("FR Y-9C (BHC)")
        report_mode = st.radio("Report Type", report_options, index=0, key="report_mode")
        is_fry = report_mode == "FR Y-9C (BHC)"

        st.divider()

        # --- Set active data based on toggle ---
        if is_fry:
            a_qoq      = fry_qoq
            a_levels    = fry_levels
            a_iso_qoq   = fry_iso_qoq
            a_iso_lvl   = fry_iso_lvl
            a_raw       = _extract_features_from_anomaly_df(fry_levels) if fry_levels else {}
            a_raw_qoq   = _extract_features_from_anomaly_df(fry_qoq) if fry_qoq else {}
            entity_label = "BHC"
        else:
            a_qoq      = ffiec_qoq
            a_levels    = ffiec_levels
            a_iso_qoq   = ffiec_iso_qoq
            a_iso_lvl   = ffiec_iso_lvl
            a_raw       = ffiec_raw
            a_raw_qoq   = ffiec_raw_qoq
            entity_label = "Bank"

        # Available models for this mode (all 4 for both report types)
        available_models = [m for m, d in [
            ('Neural Network (QoQ)',       a_qoq),
            ('Neural Network (Levels)',    a_levels),
            ('Isolation Forest (QoQ)',     a_iso_qoq),
            ('Isolation Forest (Levels)',  a_iso_lvl),
        ] if len(d) > 0]
        n_models = len(available_models)
        missing_models = [
            model_name for model_name, data_dict in [
                ('Neural Network (QoQ)', a_qoq),
                ('Neural Network (Levels)', a_levels),
                ('Isolation Forest (QoQ)', a_iso_qoq),
                ('Isolation Forest (Levels)', a_iso_lvl),
            ] if len(data_dict) == 0
        ]

        # Bank names
        all_names = set()
        for d in (a_qoq, a_levels, a_iso_qoq, a_iso_lvl):
            all_names.update(d.keys())
        bank_names = sorted(all_names)

        preferred_name = preferred_entity_name(is_fry)
        default_bank_index = bank_names.index(preferred_name) if preferred_name in bank_names else 0
        selected_bank = st.selectbox(f"Select {entity_label}", bank_names, index=default_bank_index)

        st.divider()

        # Year range
        all_years = set()
        for d in (a_qoq, a_levels, a_iso_qoq, a_iso_lvl):
            for df_tmp in d.values():
                if 'quarter' in df_tmp.columns:
                    yrs = df_tmp['quarter'].apply(parse_year).dropna().astype(int)
                    all_years.update(yrs.tolist())
        if all_years:
            min_yr, max_yr = int(min(all_years)), int(max(all_years))
            default_year_range = default_year_window_for_bank(
                selected_bank, (a_qoq, a_levels, a_iso_qoq, a_iso_lvl), min_yr, max_yr
            )
            year_range = st.slider("Year Range", min_yr, max_yr, default_year_range)
        else:
            year_range = None

        show_forecast = False
        if not is_fry:
            st.divider()
            show_forecast = st.checkbox("Show projected values (2026-2027)", value=False)
            if show_forecast:
                st.caption("Dashed = projected. Solid = reported.")

        st.caption(f"Report: **{report_mode}** | Models: **{', '.join(available_models)}**")
        if mdrm:
            st.caption(f"Line item dictionary: **{len(mdrm):,} codes**")
        if missing_models:
            st.warning("Unavailable model outputs: " + ", ".join(missing_models))
    bank_forecast_df = load_forecasts(selected_bank) if (show_forecast and not is_fry) else None
    # =========================================================================
    # YEAR FILTER
    # =========================================================================
    def _filter_years(dataframe):
        if year_range is None or 'quarter' not in dataframe.columns:
            return dataframe
        yrs = dataframe['quarter'].apply(parse_year)
        return dataframe[(yrs >= year_range[0]) & (yrs <= year_range[1])].copy()

    # =========================================================================
    # CONVERGENCE TABLE
    # =========================================================================
    # Map model names to their data dictionaries
    _model_data_map = {
        'Neural Network (QoQ)':       a_qoq,
        'Neural Network (Levels)':    a_levels,
        'Isolation Forest (QoQ)':     a_iso_qoq,
        'Isolation Forest (Levels)':  a_iso_lvl,
    }

    conv_records = []
    for bank in bank_names:
        model_flags = {}
        for model_name in available_models:
            data_dict = _model_data_map[model_name]
            flags = {}
            if bank in data_dict:
                for _, r in _filter_years(data_dict[bank]).iterrows():
                    flags[r['quarter']] = bool(r.get('is_anomaly', False))
            model_flags[model_name] = flags

        all_qs = set()
        for f in model_flags.values():
            all_qs.update(f.keys())

        for q in all_qs:
            rec = {
                'Bank':         bank,
                'Quarter':      q,
                'quarter_date': pd.to_datetime(q, format='%m/%d/%Y', errors='coerce'),
            }
            n = 0
            for m in available_models:
                val = model_flags[m].get(q, False)
                rec[m] = val
                n += val
            rec['Models Flagged'] = n
            conv_records.append(rec)

    conv_df = pd.DataFrame(conv_records)
    if not conv_df.empty:
        conv_df = conv_df.sort_values(['Bank', 'quarter_date']).reset_index(drop=True)

    # =========================================================================
    # TABS
    # =========================================================================
    tab_all, tab_bank, tab_data = st.tabs(["Quarter Monitor", "Quarter Analysis", "Raw Data Explorer"])

    # =========================================================================
    # TAB 1: ALL BANKS
    # =========================================================================
    with tab_all:
        st.subheader(f"Latest Quarter Monitor: {selected_bank}")

        selected_bank_conv = conv_df[conv_df['Bank'] == selected_bank].copy() if not conv_df.empty else pd.DataFrame()
        latest_quarter = None
        if selected_bank_conv.empty:
            st.info("No filing history is available for the selected bank in the selected year range.")
        else:
            selected_bank_conv = selected_bank_conv.sort_values('quarter_date').copy()
            latest_row = selected_bank_conv.iloc[-1]
            latest_quarter = latest_row['Quarter']
            latest_flags = [m for m in available_models if latest_row.get(m, False)]

            latest_peer = conv_df[conv_df['Quarter'] == latest_quarter].copy()
            latest_peer = latest_peer.sort_values(['Models Flagged', 'Bank'], ascending=[False, True])
            peer_rank = int((latest_peer['Models Flagged'] > latest_row['Models Flagged']).sum()) + 1
            latest_bank_raw = _filter_years(a_raw[selected_bank].copy()) if selected_bank in a_raw else None
            latest_driver_df = build_driver_table(latest_bank_raw, latest_quarter, mdrm, top_n=5)

            c1, c2, c3 = st.columns(3)
            c1.metric("Latest quarter", latest_quarter)
            if not latest_driver_df.empty:
                c2.metric("Top driver deviation", f"{latest_driver_df.iloc[0]['How Unusual']:+.2f} std")
            else:
                c2.metric("Top driver deviation", "N/A")
            c3.metric("Peer rank this quarter", f"{peer_rank} of {len(latest_peer)}")

            if latest_row['Models Flagged'] == 0:
                st.success(f"{selected_bank} was not flagged in the latest available quarter.")
            elif latest_row['Models Flagged'] == n_models:
                st.error(f"{selected_bank} was flagged by all {n_models} models in {latest_quarter}.")
            else:
                st.warning(
                    f"{selected_bank} was flagged by {int(latest_row['Models Flagged'])} of {n_models} models in {latest_quarter}."
                )

            if not latest_driver_df.empty:
                top_driver = latest_driver_df.iloc[0]
                st.markdown(f"**Primary line item this quarter:** {top_driver['Line Item']}")
                st.caption(
                    f"Deviation: **{top_driver['How Unusual']:+.2f} std devs** | "
                    + f"Value: **{top_driver['Value']:,.0f}** | "
                    + f"Recent average: **{top_driver['Average']:,.0f}**"
                )
                st.dataframe(
                    latest_driver_df[['Line Item', 'Value', 'Average', 'How Unusual']].style.format({
                        'Value': '{:,.0f}',
                        'Average': '{:,.0f}',
                        'How Unusual': '{:+.2f}',
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()
                st.subheader("Primary Line Item Over Time")
                driver_chart_df = latest_bank_raw.copy()

                flag_priority = [
                    model_name for model_name in latest_flags
                    if model_name in available_models
                ] + [m for m in available_models if m not in latest_flags]
                flag_source_map = {
                    'Neural Network (QoQ)': a_qoq,
                    'Neural Network (Levels)': a_levels,
                    'Isolation Forest (QoQ)': a_iso_qoq,
                    'Isolation Forest (Levels)': a_iso_lvl,
                }
                flag_src = None
                for model_name in flag_priority:
                    src_dict = flag_source_map.get(model_name, {})
                    candidate = src_dict.get(selected_bank)
                    if candidate is not None:
                        flag_src = _filter_years(candidate.copy())
                        break

                if flag_src is not None and 'quarter' in flag_src.columns and 'is_anomaly' in flag_src.columns:
                    driver_chart_df = driver_chart_df.merge(
                        flag_src[['quarter', 'is_anomaly']], on='quarter', how='left'
                    )
                if 'is_anomaly' not in driver_chart_df.columns:
                    driver_chart_df['is_anomaly'] = False
                else:
                    driver_chart_df['is_anomaly'] = driver_chart_df['is_anomaly'].fillna(False).astype(bool)

                _render_line_item_chart(
                    driver_chart_df,
                    top_driver['raw_code'],
                    top_driver['Line Item'],
                    mdrm,
                    highlight_quarter=latest_quarter,
                    is_levels=True,
                    mdrm_types=mdrm_types,
                    forecast_df=bank_forecast_df if not is_fry else None,
                    show_forecast=show_forecast and not is_fry,
                    chart_key=f"primary-driver-{selected_bank}-{latest_quarter}-{top_driver['raw_code']}",
                )
            else:
                st.info("No line-item driver table is available for the latest quarter in the current data view.")

            st.caption(f"Flagging models: **{format_flagged_models(latest_flags)}**")

            st.divider()
            st.subheader(f"Peer Comparison for {latest_quarter}")
            peer_display = latest_peer[['Bank', 'Models Flagged'] + [
                m for m in available_models if m in latest_peer.columns
            ]].copy()
            peer_display.rename(columns={'Bank': entity_label}, inplace=True)
            peer_display.insert(
                0,
                'Focus',
                peer_display[entity_label].map(lambda name: 'Selected' if name == selected_bank else '')
            )
            for model_name in available_models:
                if model_name in peer_display.columns:
                    peer_display[model_name] = peer_display[model_name].map({True: 'Yes', False: ''})
            st.dataframe(peer_display, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader(f"Recent Flag Trend: {selected_bank}")
            recent_trend = selected_bank_conv.tail(8).copy()
            recent_trend['bar_color'] = recent_trend['Models Flagged'].apply(
                lambda value: ANOMALY_RED if value >= 2 else ('#E9C46A' if value == 1 else 'rgba(180,180,180,0.35)')
            )

            fig_recent = go.Figure()
            fig_recent.add_trace(go.Bar(
                x=recent_trend['quarter_date'],
                y=recent_trend['Models Flagged'],
                marker_color=recent_trend['bar_color'],
                customdata=recent_trend[['Quarter']].values,
                hovertemplate='%{customdata[0]}<br>%{y} of ' + str(n_models) + ' models flagged<extra></extra>',
            ))
            fig_recent.update_layout(**plotly_theme_layout())
            fig_recent.update_layout(
                yaxis_title="Models flagging this quarter",
                height=340,
                showlegend=False,
            )
            fig_recent.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_recent.update_yaxes(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                range=[0, n_models + 0.5],
                dtick=1,
                tickvals=list(range(n_models + 1)),
            )
            st.plotly_chart(fig_recent, use_container_width=True)

        # --- Convergence tables (all report types) ---
        st.divider()
        st.subheader("High-Confidence Flags (2+ Models)")
        hc = conv_df[conv_df['Models Flagged'] >= 2].copy() if not conv_df.empty else pd.DataFrame()
        if not hc.empty:
            hc = hc.sort_values(['Models Flagged', 'quarter_date'], ascending=[False, False])
            disp = hc[['Bank', 'Quarter', 'Models Flagged'] + [m for m in available_models if m in hc.columns]].copy()
            for m in available_models:
                if m in disp.columns:
                    disp[m] = disp[m].map({True: 'Yes', False: ''})
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("No quarters flagged by 2+ models in the selected year range.")

        st.divider()
        st.subheader("Flag Summary by " + entity_label)
        summary_rows = []
        for bank in bank_names:
            bc = conv_df[conv_df['Bank'] == bank] if not conv_df.empty else pd.DataFrame()
            row = {entity_label: bank}
            for m in available_models:
                row[m] = int(bc[m].sum()) if (len(bc) > 0 and m in bc.columns) else 0
            row['Any Model'] = int((bc['Models Flagged'] >= 1).sum()) if len(bc) > 0 else 0
            row['2+ Models'] = int((bc['Models Flagged'] >= 2).sum()) if len(bc) > 0 else 0
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # --- Flagged Line Items ---
        st.divider()
        st.subheader("Flagged Line Items")
        st.caption(
            "Each row is a line item that deviated significantly from that "
            + entity_label.lower()
            + "'s own historical average during a flagged quarter. "
            "Sorted by severity."
        )

        col_basis, col_thresh = st.columns(2)
        with col_basis:
            li_basis = st.radio(
                "Measure basis",
                ["Levels (absolute values)", "QoQ (quarter-over-quarter changes)"],
                index=0,
                horizontal=True,
                key="li_basis_allbanks",
            )
        with col_thresh:
            z_threshold = st.slider(
                "Deviation threshold (std devs)", 1.5, 4.0, 2.0, 0.25,
                key="z_thresh_allbanks",
            )
        li_raw = a_raw if li_basis.startswith("Levels") else a_raw_qoq

        if not li_raw:
            if not is_fry:
                source_path = FEATURES_RAW_DIR if li_basis.startswith("Levels") else QOQ_RAW_DIR
                source_label = "FFIEC levels" if li_basis.startswith("Levels") else "FFIEC quarter-over-quarter"
                st.warning(source_warning_text(source_label, source_path))
            else:
                st.warning("Line-item source data is unavailable for the selected basis.")
        else:
            z_df = compute_cross_bank_zscores(li_raw, conv_df, threshold=z_threshold)

            if not z_df.empty:
                flagged_tbl = z_df.copy()
                flagged_tbl['Line Item'] = [
                    translate_code(c, mdrm) for c in flagged_tbl['feature_code']
                ]
                flagged_tbl[entity_label] = flagged_tbl['bank']
                flagged_tbl['Quarter'] = flagged_tbl['quarter_date'].dt.strftime('%m/%d/%Y')
                flagged_tbl['abs_z'] = flagged_tbl['z_score'].abs()
                flagged_tbl['Deviation'] = flagged_tbl['z_score'].apply(
                    lambda z: f"+{z:.1f} std" if z > 0 else f"{z:.1f} std"
                )
                flagged_tbl = flagged_tbl.sort_values('abs_z', ascending=False)
                st.dataframe(
                    flagged_tbl[['Line Item', entity_label, 'Quarter', 'Deviation']],
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )
            else:
                st.info("No line items exceed the deviation threshold in flagged quarters.")

    # =========================================================================
    # TAB 2: BANK DETAIL
    # =========================================================================
    with tab_bank:
        st.subheader(f"Quarter Analysis: {selected_bank}")

        bank_iso_qoq = _filter_years(a_iso_qoq[selected_bank].copy()) if selected_bank in a_iso_qoq else None
        bank_iso_lvl = _filter_years(a_iso_lvl[selected_bank].copy()) if selected_bank in a_iso_lvl else None

        bank_raw = None
        if selected_bank in a_raw:
            bank_raw = _filter_years(a_raw[selected_bank].copy())

        bank_conv = conv_df[conv_df['Bank'] == selected_bank].copy() if not conv_df.empty else pd.DataFrame()

        n_total = len(bank_conv)
        n_any = int((bank_conv['Models Flagged'] >= 1).sum()) if len(bank_conv) > 0 else 0
        n_high = int((bank_conv['Models Flagged'] >= 2).sum()) if len(bank_conv) > 0 else 0

        st.subheader("Quarter Drill-Down")
        ordered_qs = ordered_quarters_for_detail(bank_conv)

        if not ordered_qs:
            st.info("No quarter data available.")
        else:
            selected_quarter = st.selectbox(
                "Select a quarter to inspect", ordered_qs, index=0, key="drill_down_quarter"
            )
            q_row = bank_conv[bank_conv['Quarter'] == selected_quarter].sort_values('quarter_date').iloc[-1]
            flags = [m for m in available_models if q_row.get(m, False)]
            n_flags = int(q_row['Models Flagged'])

            c1, c2, c3 = st.columns(3)
            c1.metric("Quarter under review", selected_quarter)
            c2.metric("Models flagged", f"{n_flags} of {n_models}")
            c3.metric("High-confidence quarters", n_high)

            if n_flags == 0:
                st.info(f"**{selected_quarter}** - No model flagged this quarter.")
            elif n_flags == n_models:
                st.error(f"**{selected_quarter}** - Flagged by all {n_models} models.")
            else:
                st.warning(f"**{selected_quarter}** - Flagged by {n_flags} model(s).")

            st.caption(f"Flagging models: **{format_flagged_models(flags)}**")

            if bank_iso_qoq is not None and 'quarter' in bank_iso_qoq.columns:
                iso_q = bank_iso_qoq[bank_iso_qoq['quarter'] == selected_quarter]
                if len(iso_q) > 0 and 'systemic_score' in iso_q.columns:
                    r = iso_q.iloc[0]
                    st.caption(
                        "Isolation Forest (QoQ) | "
                        + f"Overall anomaly score: **{r['anom_score']:.4f}** | "
                        + f"Peer-wide movement: **{r['systemic_score']:.4f}** | "
                        + f"{entity_label}-specific excess: **{r['adjusted_score']:.4f}**"
                    )
            if bank_iso_lvl is not None and 'quarter' in bank_iso_lvl.columns:
                iso_q = bank_iso_lvl[bank_iso_lvl['quarter'] == selected_quarter]
                if len(iso_q) > 0 and 'systemic_score' in iso_q.columns:
                    r = iso_q.iloc[0]
                    st.caption(
                        "Isolation Forest (Levels) | "
                        + f"Overall anomaly score: **{r['anom_score']:.4f}** | "
                        + f"Peer-wide movement: **{r['systemic_score']:.4f}** | "
                        + f"{entity_label}-specific excess: **{r['adjusted_score']:.4f}**"
                    )

            if bank_raw is not None and selected_quarter in bank_raw['quarter'].values:
                feat_skip = {'quarter', 'quarter_date'}
                feat_cols = [c for c in bank_raw.columns if c not in feat_skip and not c.startswith('_')]
                feat_data = bank_raw[feat_cols].apply(pd.to_numeric, errors='coerce')
                bank_means = feat_data.mean()
                bank_stds = feat_data.std().replace(0, np.nan)
                row_vals = feat_data[bank_raw['quarter'] == selected_quarter].iloc[0]
                z_scores = ((row_vals - bank_means) / bank_stds).dropna()

                n_top = st.slider("Top line items to show", 5, 30, 15, key="n_top_features")
                top_features = z_scores.abs().nlargest(n_top)

                top_df = pd.DataFrame({
                    'raw_code': top_features.index,
                    'Line Item': [translate_code(f, mdrm) for f in top_features.index],
                    'Chart Label': [chart_label(f, mdrm) for f in top_features.index],
                    'Value': [row_vals[f] for f in top_features.index],
                    'Average': [bank_means[f] for f in top_features.index],
                    'How Unusual': [z_scores[f] for f in top_features.index],
                })

                colors = [ANOMALY_RED if z > 0 else NORMAL_BLUE for z in top_df['How Unusual']]
                fig_drivers = go.Figure()
                fig_drivers.add_trace(go.Bar(
                    x=top_df['How Unusual'],
                    y=top_df['Chart Label'],
                    orientation='h',
                    marker_color=colors,
                    hovertemplate=(
                        '%{customdata[2]}<br>'
                        'How unusual: %{x:.1f} std devs<br>'
                        'Value: %{customdata[0]:,.0f}<br>'
                        'Avg: %{customdata[1]:,.0f}<extra></extra>'
                    ),
                    customdata=top_df[['Value', 'Average', 'Line Item']].values,
                ))
                fig_drivers.update_layout(**plotly_theme_layout())
                fig_drivers.update_layout(
                    xaxis_title="Standard deviations from average",
                    height=max(350, n_top * 30),
                    yaxis=dict(autorange='reversed'),
                )
                fig_drivers.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                fig_drivers.update_yaxes(showgrid=False)
                st.plotly_chart(fig_drivers, use_container_width=True)

                st.dataframe(
                    top_df[['Line Item', 'Value', 'Average', 'How Unusual']].style.format({
                        'Value': '{:,.0f}',
                        'Average': '{:,.0f}',
                        'How Unusual': '{:.2f}',
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()
                st.subheader("Line Item Over Time")
                li_options = [chart_label(c, mdrm) for c in top_df['raw_code']]
                display_to_raw = {chart_label(c, mdrm): c for c in top_df['raw_code']}

                selected_li = st.selectbox(
                    "Select line item to graph", li_options, index=0, key="li_drilldown"
                )
                if selected_li:
                    raw_code = display_to_raw[selected_li]
                    chart_df = bank_raw.copy()
                    flag_src = a_iso_qoq.get(selected_bank)
                    if flag_src is None:
                        flag_src = a_qoq.get(selected_bank)
                    if flag_src is None:
                        flag_src = a_levels.get(selected_bank)
                    if flag_src is not None:
                        chart_df = chart_df.merge(
                            flag_src[['quarter', 'is_anomaly']], on='quarter', how='left'
                        )
                    if 'is_anomaly' not in chart_df.columns:
                        chart_df['is_anomaly'] = False
                    else:
                        chart_df['is_anomaly'] = chart_df['is_anomaly'].fillna(False).astype(bool)

                    _render_line_item_chart(
                        chart_df, raw_code, selected_li, mdrm,
                        highlight_quarter=selected_quarter,
                        is_levels=True,
                        mdrm_types=mdrm_types,
                        forecast_df=None,
                        show_forecast=False,
                        chart_key=f"analysis-driver-{selected_bank}-{selected_quarter}-{raw_code}",
                    )
            elif bank_raw is None:
                if not is_fry:
                    st.warning(source_warning_text("FFIEC levels", FEATURES_RAW_DIR))
                else:
                    st.warning("Raw features data is not available for line-item analysis.")
            else:
                st.warning(f"Quarter {selected_quarter} was not found in the available features data.")

        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total quarters in view", n_total)
        c2.metric("Flagged by any model", n_any)
        c3.metric("High confidence (2+ models)", n_high)

        st.divider()
        st.subheader("Flagged Quarters Over Time")
        if len(bank_conv) > 0:
            bc_chart = bank_conv.sort_values('quarter_date').copy()
            color_map = {0: 'rgba(180,180,180,0.15)', 1: '#E9C46A', 2: '#F4842A', 3: ANOMALY_RED, 4: ANOMALY_RED}
            bc_chart['bar_color'] = bc_chart['Models Flagged'].map(color_map)

            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(
                x=bc_chart['quarter_date'],
                y=bc_chart['Models Flagged'],
                marker_color=bc_chart['bar_color'],
                hovertemplate='%{customdata[0]}<br>%{y} of ' + str(n_models) + ' models flagged<extra></extra>',
                customdata=bc_chart[['Quarter']].values,
            ))
            fig_bars.update_layout(**plotly_theme_layout())
            fig_bars.update_layout(
                yaxis_title="Models flagging this quarter", height=350,
                showlegend=False, bargap=0.1,
            )
            fig_bars.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_bars.update_yaxes(
                showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                range=[0, n_models + 0.5], dtick=1, tickvals=list(range(n_models + 1)),
            )
            st.plotly_chart(fig_bars, use_container_width=True)

        st.divider()
        st.subheader("Flagged Quarters")
        if len(bank_conv) > 0:
            flagged_bank = bank_conv[bank_conv['Models Flagged'] >= 1].sort_values(
                ['quarter_date', 'Models Flagged'], ascending=[False, False]
            )[['Quarter', 'Models Flagged'] + [m for m in available_models if m in bank_conv.columns]].copy()
            if len(flagged_bank) > 0:
                for m in available_models:
                    if m in flagged_bank.columns:
                        flagged_bank[m] = flagged_bank[m].map({True: 'Yes', False: ''})
                st.dataframe(flagged_bank, use_container_width=True, hide_index=True)

        diagnostics_available = any(not df.empty for df in [iso_lvl_findings, iso_lvl_stability, iso_lvl_precision])
        if not is_fry and diagnostics_available:
            st.divider()
            with st.expander("Advanced diagnostics", expanded=False):
                st.caption(
                    "These tables are advanced diagnostic artifacts for the canonical FFIEC Isolation Forest levels outputs. "
                    + "Use them as supporting detail, not as the primary quarter-triage signal."
                )

                bank_findings = iso_lvl_findings[iso_lvl_findings['bank_name'] == selected_bank] if not iso_lvl_findings.empty else pd.DataFrame()
                bank_stability = iso_lvl_stability[iso_lvl_stability['bank_name'] == selected_bank] if not iso_lvl_stability.empty else pd.DataFrame()
                bank_precision = iso_lvl_precision[iso_lvl_precision['bank_name'] == selected_bank] if not iso_lvl_precision.empty else pd.DataFrame()

                if not bank_findings.empty:
                    st.markdown("**Primary Findings**")
                    disp_f = bank_findings[['quarter', 'anom_score', 'systemic_score', 'adjusted_score']].copy()
                    disp_f.columns = ['Quarter', 'Overall anomaly score', 'Peer-wide movement', 'Bank-specific excess']
                    disp_f = disp_f.sort_values('Bank-specific excess', ascending=False)
                    st.dataframe(disp_f.style.format({
                        'Overall anomaly score': '{:.4f}',
                        'Peer-wide movement': '{:.4f}',
                        'Bank-specific excess': '{:.4f}',
                    }), use_container_width=True, hide_index=True)

                if not bank_stability.empty:
                    st.markdown("**Multi-Seed Stability**")
                    stab_cols = [c for c in ['quarter', 'mean_score', 'std_score', 'seeds_flagged', 'flag_rate', 'volatile'] if c in bank_stability.columns]
                    disp_s = bank_stability[stab_cols].rename(columns={
                        'quarter': 'Quarter',
                        'mean_score': 'Mean score',
                        'std_score': 'Std dev',
                        'seeds_flagged': 'Seeds flagged',
                        'flag_rate': 'Flag rate',
                        'volatile': 'Stability',
                    })
                    if 'Stability' in disp_s.columns:
                        disp_s['Stability'] = disp_s['Stability'].fillna('Stable')
                    fmt = {}
                    if 'Mean score' in disp_s.columns:
                        fmt['Mean score'] = '{:.4f}'
                    if 'Std dev' in disp_s.columns:
                        fmt['Std dev'] = '{:.4f}'
                    if 'Flag rate' in disp_s.columns:
                        fmt['Flag rate'] = '{:.0%}'
                    st.dataframe(disp_s.style.format(fmt), use_container_width=True, hide_index=True)

                if not bank_precision.empty:
                    st.markdown("**Precision Audit**")
                    prec_cols = [c for c in bank_precision.columns if c != 'bank_name']
                    disp_p = bank_precision[prec_cols].rename(columns={
                        'quarter': 'Quarter',
                        'adjusted_score': 'Bank-specific excess',
                        'key': 'Review key',
                    })
                    fmt_p = {'Bank-specific excess': '{:.4f}'} if 'Bank-specific excess' in disp_p.columns else {}
                    st.dataframe(disp_p.style.format(fmt_p), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: DATA VIEWER
    # =========================================================================
    with tab_data:
        st.subheader("Raw Data Explorer")

        data_source = st.radio(
            "Data type",
            ["Absolute Values (Levels)", "Quarter-over-Quarter Changes (QoQ)"],
            horizontal=True, key="data_source_radio",
        )
        is_qoq = "QoQ" in data_source

        raw_banks = a_raw_qoq if is_qoq else a_raw
        value_label = "QoQ Change (%)" if is_qoq else "Value"

        if not raw_banks:
            if not is_fry:
                source_path = FEATURES_RAW_DIR if not is_qoq else QOQ_RAW_DIR
                source_label = "FFIEC levels" if not is_qoq else "FFIEC quarter-over-quarter"
                st.warning(source_warning_text(source_label, source_path))
            else:
                st.warning("No data files found for this report type / data type.")
        elif selected_bank not in raw_banks:
            st.warning(f"No raw data for {selected_bank}.")
        else:
            rdf = _filter_years(raw_banks[selected_bank].copy())
            data_cols = [c for c in rdf.columns if c not in ('quarter', 'quarter_date')]

            if not data_cols:
                st.warning("No line items are available for the selected view.")
            else:
                viewer_options = []
                viewer_raw_map = {}
                for code in sorted(data_cols):
                    display = chart_label(code, mdrm)
                    viewer_options.append(display)
                    viewer_raw_map[display] = code

                selected_viewer_item = st.selectbox(
                    "Search for a line item", viewer_options, index=0, key="viewer_line_item"
                )
                st.caption(f"Full line item: {translate_code(viewer_raw_map[selected_viewer_item], mdrm, truncate=False)}")

            if data_cols and selected_viewer_item:
                raw_code  = viewer_raw_map[selected_viewer_item]
                unit      = get_unit_label(raw_code, mdrm_types)
                unit_suffix = f" ({unit})" if unit else ""

                vals      = pd.to_numeric(rdf[raw_code], errors='coerce')
                col_label = f"{value_label}{unit_suffix}"
                table_df  = pd.DataFrame({'Quarter': rdf['quarter'], col_label: vals})
                delta_label = "Change from Prior Quarter" if not is_qoq else "Change in QoQ Change"
                table_df[delta_label] = vals.diff()

                st.divider()
                st.subheader(f"{_truncate(short_name(raw_code, mdrm), 60)} Over Time")
                if unit:
                    st.caption(f"Unit: **{unit}**")

                compatible_compare_banks = [
                    b for b in bank_names
                    if b != selected_bank and b in raw_banks and raw_code in raw_banks[b].columns
                ]
                compare_options = ["None"] + compatible_compare_banks
                if st.session_state.get("viewer_compare_bank") not in compare_options:
                    st.session_state["viewer_compare_bank"] = "None"
                compare_viewer = st.selectbox(
                    "Overlay another " + entity_label.lower(),
                    compare_options,
                    key="viewer_compare_bank",
                )
                if compatible_compare_banks and len(compatible_compare_banks) < len(bank_names) - 1:
                    st.caption("Only banks that report this line item are shown for overlay.")
                elif not compatible_compare_banks:
                    st.caption("No other banks report this line item in the raw data.")

                compare_rdf = None
                compare_vals = None
                if compare_viewer != "None" and compare_viewer in raw_banks:
                    compare_rdf = _filter_years(raw_banks[compare_viewer].copy())
                    compare_vals = pd.to_numeric(compare_rdf[raw_code], errors='coerce')

                primary_trace_name = selected_bank if compare_vals is not None else col_label
                fig_viewer = go.Figure()
                fig_viewer.add_trace(go.Scatter(
                    x=rdf['quarter_date'], y=vals,
                    mode='lines+markers', name=primary_trace_name,
                    marker=dict(color=NORMAL_BLUE, size=5),
                    line=dict(color=NORMAL_BLUE, width=2),
                    hovertemplate='%{customdata[0]}<br>' + col_label + ': %{y:,.2f}<extra></extra>',
                    customdata=rdf[['quarter']].values,
                ))

                if compare_vals is not None:
                    fig_viewer.add_trace(go.Scatter(
                        x=compare_rdf['quarter_date'], y=compare_vals,
                        mode='lines+markers', name=compare_viewer,
                        marker=dict(color=NN_GREEN, size=5),
                        line=dict(color=NN_GREEN, width=2, dash='dash'),
                        hovertemplate='%{customdata[0]}<br>' + col_label + ': %{y:,.2f}<extra></extra>',
                        customdata=compare_rdf[['quarter']].values,
                    ))

                # Forecast overlay (FFIEC levels only)
                if show_forecast and not is_qoq and not is_fry and bank_forecast_df is not None:
                    fc = bank_forecast_df[bank_forecast_df['feature'] == raw_code].copy()
                    if not fc.empty:
                        last_row = rdf.dropna(subset=['quarter_date']).sort_values('quarter_date').iloc[-1]
                        last_val = pd.to_numeric(last_row[raw_code], errors='coerce')
                        connector = pd.DataFrame([{
                            'quarter_date': last_row['quarter_date'],
                            'quarter': last_row['quarter'],
                            'value': last_val,
                        }])
                        fc_plot = pd.concat([connector, fc[['quarter_date', 'quarter', 'value']]], ignore_index=True)
                        fig_viewer.add_trace(go.Scatter(
                            x=fc_plot['quarter_date'], y=fc_plot['value'],
                            mode='lines', name='Projected (2026-2027)',
                            line=dict(color=NORMAL_BLUE, width=2, dash='dash'),
                            opacity=0.65,
                        ))
                        fig_viewer.add_vline(
                            x=FORECAST_CUTOFF.timestamp() * 1000,
                            line_dash='dot', line_color='rgba(150,150,150,0.6)',
                            annotation_text='Projections ->',
                            annotation_position='top right', annotation_font_size=11,
                        )

                fig_viewer.update_layout(**plotly_theme_layout())
                fig_viewer.update_layout(
                    yaxis_title=col_label, height=400,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                )
                fig_viewer.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                fig_viewer.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                st.plotly_chart(fig_viewer, use_container_width=True)

                st.divider()
                st.subheader("Values by Quarter")
                st.dataframe(
                    table_df.style.format({col_label: '{:,.2f}', delta_label: '{:,.2f}'}),
                    use_container_width=True, hide_index=True,
                )

def _render_line_item_chart(df, raw_code, display_name, mdrm,
                            highlight_quarter=None, is_levels=False,
                            mdrm_types=None, forecast_df=None, show_forecast=False,
                            chart_key=None):
    if raw_code not in df.columns or 'quarter_date' not in df.columns:
        st.warning("Unable to render chart - data not available.")
        return

    values = pd.to_numeric(df[raw_code], errors='coerce')

    unit = get_unit_label(raw_code, mdrm_types or {})
    y_label = (f"Value ({unit})" if unit else "Value") if is_levels else \
              (f"QoQ Change ({unit})" if unit else "Quarter-over-Quarter Change (%)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['quarter_date'], y=values,
        mode='lines+markers', name='Reported values',
        marker=dict(color=NORMAL_BLUE, size=5),
        line=dict(color=NORMAL_BLUE, width=2),
        hovertemplate='%{customdata[0]}<br>' + y_label + ': %{y:,.2f}<extra></extra>',
        customdata=df[['quarter']].values,
    ))

    if highlight_quarter:
        hq = df[df['quarter'] == highlight_quarter]
        if len(hq) > 0:
            hq_val = pd.to_numeric(hq[raw_code].iloc[0], errors='coerce')
            fig.add_trace(go.Scatter(
                x=hq['quarter_date'], y=[hq_val],
                mode='markers', name=f'Selected ({highlight_quarter})',
                marker=dict(color='#FFD700', size=16, symbol='star',
                            line=dict(width=2, color='black')),
            ))

    if show_forecast and forecast_df is not None:
        fc = forecast_df[forecast_df['feature'] == raw_code].copy()
        if not fc.empty:
            last_hist = df.dropna(subset=['quarter_date']).sort_values('quarter_date').iloc[-1]
            last_val  = pd.to_numeric(last_hist[raw_code], errors='coerce')
            connector = pd.DataFrame([{
                'quarter_date': last_hist['quarter_date'],
                'quarter': last_hist['quarter'],
                'value': last_val,
            }])
            fc_plot = pd.concat([connector, fc[['quarter_date', 'quarter', 'value']]], ignore_index=True)
            fig.add_trace(go.Scatter(
                x=fc_plot['quarter_date'], y=fc_plot['value'],
                mode='lines', name='Projected (2026-2027)',
                line=dict(color=NORMAL_BLUE, width=1.5, dash='dash'),
                opacity=0.65,
            ))
            fig.add_vline(
                x=FORECAST_CUTOFF.timestamp() * 1000,
                line_dash='dot', line_color='rgba(150,150,150,0.6)',
                annotation_text='Projections ->',
                annotation_position='top right', annotation_font_size=11,
            )

    fig.update_layout(**plotly_theme_layout())
    fig.update_layout(
        title=dict(text=short_name(raw_code, mdrm), y=0.98, yanchor='top'),
        yaxis_title=y_label, height=430,
        margin=dict(t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


if __name__ == "__main__":
    main()

