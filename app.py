import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from features import prepare_model_input

st.set_page_config(
    page_title="Titanium Alloy Modulus Specification | Materials Informatics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Editorial Typography & Organic Aesthetics -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Font Overrides */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #dedbd2;
    -webkit-font-smoothing: antialiased;
}

/* Headings in Editorial Serif */
h1, h2, h3, .editorial-heading {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
    color: #f3efe6 !important;
}

/* Monospace Figures & Data */
.mono-figure {
    font-family: 'JetBrains Mono', monospace;
    font-feature-settings: 'tnum' 1;
}

/* Header Masthead */
.masthead-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9c9586;
    margin-bottom: 0.35rem;
}
.masthead-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.35rem;
    font-weight: 500;
    line-height: 1.15;
    color: #f7f4ee;
    margin: 0;
}
.masthead-subdeck {
    font-size: 0.95rem;
    color: #a49e91;
    line-height: 1.6;
    max-width: 860px;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}
.hairline-rule {
    border: none;
    border-top: 1px solid rgba(220, 215, 205, 0.14);
    margin: 1.25rem 0 1.75rem 0;
}

/* Editorial Cards & Sections */
.editorial-card {
    background-color: #17181a;
    border: 1px solid rgba(220, 215, 205, 0.12);
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #928b7e;
    margin-bottom: 0.6rem;
}

/* Primary Button Overrides */
div.stButton > button {
    background-color: #c46849 !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 3px !important;
    border: 1px solid #b2583a !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.15s ease-in-out !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
}
div.stButton > button:hover {
    background-color: #d17556 !important;
    border-color: #c46849 !important;
    transform: translateY(-1px);
}
div.stButton > button:active {
    transform: translateY(0);
}

/* Structured Prediction Box */
.prediction-box {
    background: #191b1e;
    border: 1px solid rgba(220, 215, 205, 0.22);
    border-radius: 3px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.8rem;
    margin-bottom: 1.25rem;
}
.prediction-box-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aba497;
    margin-bottom: 0.8rem;
}
.prediction-box-value {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 3.1rem;
    font-weight: 500;
    line-height: 1.05;
    color: #f7f4ee;
    margin-bottom: 1rem;
}
.prediction-box-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    color: #a49e91;
    margin-left: 0.4rem;
}
.prediction-box-meta {
    border-top: 1px solid rgba(220, 215, 205, 0.12);
    padding-top: 0.85rem;
    font-size: 0.86rem;
    color: #c9c5bc;
    line-height: 1.65;
}

/* Model Performance & Validation Section */
.bench-section {
    background: #161719;
    border: 1px solid rgba(220, 215, 205, 0.12);
    border-radius: 3px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
}
.bench-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 500;
    color: #f3efe6;
    margin-bottom: 0.25rem;
}
.bench-divider {
    border-top: 1px solid rgba(220, 215, 205, 0.16);
    margin: 0.45rem 0 0.85rem 0;
}
.bench-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.22rem 0;
    color: #d1cdc4;
}
.bench-row-label {
    color: #928b7e;
}
.bench-row-value {
    color: #f0ece3;
    font-weight: 500;
}

/* Caution Note */
.caution-note {
    background: rgba(196, 104, 73, 0.08);
    border-left: 3px solid #c46849;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #d99a84;
    line-height: 1.5;
    margin-top: 1rem;
    border-radius: 0 2px 2px 0;
}

/* Editorial Footer */
.editorial-footer {
    border-top: 1px solid rgba(220, 215, 205, 0.12);
    padding: 2rem 0 1.5rem 0;
    margin-top: 3rem;
    font-size: 0.8rem;
    color: #7b7569;
    line-height: 1.7;
}

/* Dataframe & Table Refinements */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(220, 215, 205, 0.12);
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Load Model & Metrics -----------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'ti_ym_model_v2.joblib')

@st.cache_resource
def load_feature_columns():
    with (BASE_DIR / 'feature_columns_v2.json').open(encoding='utf-8') as file:
        return json.load(file)

@st.cache_resource
def load_training_columns():
    return pd.read_csv(BASE_DIR / 'train_v2.csv', nrows=0).columns.tolist()

@st.cache_resource
def load_metrics():
    with (BASE_DIR / 'results_v2.json').open(encoding='utf-8') as file:
        results = json.load(file)
    selected = results['selected_model']
    cv = results['cross_validation']['metrics'][selected['label']]
    return {
        "selected_model": selected['label'],
        "cv_r2_mean": cv['r2_mean'],
        "cv_mae_gpa": cv['mae_mean'],
        "cv_rmse_gpa": cv['rmse_mean'],
        "test_r2": results['test_metrics']['r2'],
        "test_mae_gpa": results['test_metrics']['mae'],
        "test_rmse_gpa": results['test_metrics']['rmse'],
        "training_rows": results['split']['statistics']['train']['rows'],
        "test_rows": results['split']['statistics']['test']['rows'],
    }

FEATURES = load_feature_columns()
TRAINING_COLUMNS = load_training_columns()
COMPOSITION_COLUMNS = [
    column for column in TRAINING_COLUMNS if column.endswith('_wt')
]
metrics = load_metrics()

# ----------------- Session State Initialization -----------------
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'predicted_alloy' not in st.session_state:
    st.session_state['predicted_alloy'] = None
if 'predicted_ti' not in st.session_state:
    st.session_state['predicted_ti'] = None
if 'predicted_mo_eq' not in st.session_state:
    st.session_state['predicted_mo_eq'] = None
if 'predicted_vec' not in st.session_state:
    st.session_state['predicted_vec'] = None
if 'uncertainty' not in st.session_state:
    st.session_state['uncertainty'] = None
if 'confidence' not in st.session_state:
    st.session_state['confidence'] = None
if 'exp_val' not in st.session_state:
    st.session_state['exp_val'] = None

element_defaults = {
    'Nb': 10.0, 'Zr': 5.0, 'Fe': 1.0, 'Sn': 4.0,
    'Ta': 0.0, 'Mo': 0.0, 'Al': 0.0, 'V': 0.0,
    'Cr': 0.0, 'Cu': 0.0, 'W': 0.0, 'Co': 0.0,
    'Hf': 0.0, 'Mn': 0.0, 'O': 0.0, 'Pd': 0.0,
    'Sc': 0.0, 'Si': 0.0,
}
for elem, val in element_defaults.items():
    if f'val_{elem}' not in st.session_state:
        st.session_state[f'val_{elem}'] = float(val)

# ----------------- Editorial Masthead -----------------
head_col1, head_col2 = st.columns([1.5, 0.5])
with head_col1:
    st.markdown('<div class="masthead-kicker">Materials Informatics Laboratory // Specification Framework</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="masthead-title">Elastic Modulus & Phase Equilibrium Engine</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="masthead-subdeck">'
        "High-fidelity computational estimation of Young's Modulus (E) across multicomponent titanium alloys. "
        'Evaluated via empirical valence electron concentrations, Molybdenum equivalence, and ensemble regression.'
        '</div>',
        unsafe_allow_html=True
    )
with head_col2:
    st.markdown('<div style="margin-top: 1.6rem;"></div>', unsafe_allow_html=True)
    with st.popover("Terminology & Metrics Guide", use_container_width=True):
        cv_r2 = metrics.get("cv_r2_mean", float('nan'))
        cv_mae = metrics.get("cv_mae_gpa", float('nan'))
        cv_rmse = metrics.get("cv_rmse_gpa", float('nan'))
        st.markdown(f"""
        ### Terminology Guide
        
        **Young's Modulus (E)**  
        The measure of a material's elastic stiffness (in GPa). In biomedical bone implants, a low modulus (< 60 GPa) prevents stress shielding. In aerospace structures, a high modulus (> 100 GPa) provides rigid load bearing.
        
        **Estimated Uncertainty (± GPa)**  
        The statistical margin of error derived from the model's cross-validation residuals (MAE).
        
        **Confidence Level**  
        Qualitative reliability rating (High, Moderate, Low) based on how well the input composition falls within the training data distribution.
        
        **5-Fold Cross-Validation (CV)**  
        A validation procedure where data is split into 5 folds. The model is trained on 4 and tested on the held-out fold, rotated 5 times to prevent overfitting.
        
        **R² Score ({cv_r2:.2f})**  
        Coefficient of determination measuring the proportion of variance in Young's modulus explained by compositional descriptors.
        
        **MAE ({cv_mae:.2f} GPa) & RMSE ({cv_rmse:.2f} GPa)**  
        Mean Absolute Error and Root Mean Squared Error, quantifying average prediction deviation across experimental alloys.
        
        **External Validation & Error (%)**  
        Direct percentage deviation between user-entered laboratory measurements and model predictions:  
        $$\\\\text{{Error}} = \\\\frac{{|\\\\text{{Predicted}} - \\\\text{{Experimental}}|}}{{\\\\text{{Experimental}}}} \\\\times 100\\\\%$$
        
        **Processing Conditions Note**  
        Measured modulus varies based on heat treatment, solution treating, aging, and cold working, which alter $\\\\beta$, $\\\\alpha$, and $\\\\omega$ phase fractions.
        """)

st.markdown('<div class="hairline-rule"></div>', unsafe_allow_html=True)

# ----------------- Sidebar Formulation Controls -----------------
st.sidebar.markdown('<div class="card-label">Specification Panel</div>', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="margin-top: 0; font-size: 1.35rem;">Alloy Formulation (wt. %)</h3>', unsafe_allow_html=True)

input_mode = st.sidebar.radio(
    "Specification Method",
    ["Direct Numeric Input", "Proportional Sliders"],
    index=0,
    label_visibility="collapsed"
)

def render_element_input(elem_key, label, step=0.1):
    current_val = float(st.session_state.get(f'val_{elem_key}', 0.0))
    if input_mode == "Direct Numeric Input":
        val = st.sidebar.number_input(
            label,
            min_value=0.0,
            max_value=100.0,
            value=current_val,
            step=step,
            format="%.2f",
            key=f"num_{elem_key}"
        )
    else:
        val = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value=100.0,
            value=current_val,
            step=step,
            format="%.2f",
            key=f"slider_{elem_key}"
        )
    st.session_state[f'val_{elem_key}'] = float(val)
    return float(val)

st.sidebar.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8f887b; margin: 1rem 0 0.4rem 0;">Primary Solutes</div>', unsafe_allow_html=True)
nb = render_element_input('Nb', "Niobium (Nb)", step=0.5)
zr = render_element_input('Zr', "Zirconium (Zr)", step=0.5)
fe = render_element_input('Fe', "Iron (Fe)", step=0.1)
sn = render_element_input('Sn', "Tin (Sn)", step=0.5)

with st.sidebar.expander("Secondary Solutes"):
    ta = render_element_input('Ta', "Tantalum (Ta)", step=0.5)
    mo = render_element_input('Mo', "Molybdenum (Mo)", step=0.5)
    al = render_element_input('Al', "Aluminium (Al)", step=0.5)
    v  = render_element_input('V', "Vanadium (V)", step=0.5)
    cr = render_element_input('Cr', "Chromium (Cr)", step=0.5)
    cu = render_element_input('Cu', "Copper (Cu)", step=0.5)
    w  = render_element_input('W', "Tungsten (W)", step=0.5)
    co = render_element_input('Co', "Cobalt (Co)", step=0.5)
    hf = render_element_input('Hf', "Hafnium (Hf)", step=0.5)
    mn = render_element_input('Mn', "Manganese (Mn)", step=0.5)
    oxygen = render_element_input('O', "Oxygen (O)", step=0.1)
    palladium = render_element_input('Pd', "Palladium (Pd)", step=0.5)
    sc = render_element_input('Sc', "Scandium (Sc)", step=0.5)
    si = render_element_input('Si', "Silicon (Si)", step=0.5)

alloy_elements = {
    'Nb': nb, 'Zr': zr, 'Fe': fe, 'Sn': sn,
    'Ta': ta, 'Mo': mo, 'Al': al, 'V': v,
    'Cr': cr, 'Cu': cu, 'W': w, 'Co': co,
    'Hf': hf, 'Mn': mn, 'O': oxygen, 'Pd': palladium,
    'Sc': sc, 'Si': si,
}

sum_other = round(sum(alloy_elements.values()), 4)

# Mass Balance Logic
if sum_other > 100.0:
    ti = 0.0
    total_comp = sum_other
    is_valid_composition = False
else:
    ti = round(100.0 - sum_other, 2)
    total_comp = round(ti + sum_other, 2)
    is_valid_composition = True

# Feature calculations use the same function and tables used for dataset generation.
beta_stabilizers = nb + zr + fe + mo

st.sidebar.markdown('<div class="hairline-rule" style="margin: 1.2rem 0;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="card-label">Formulation Balance</div>', unsafe_allow_html=True)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    st.metric("Titanium Matrix", f"{ti:.2f}%")
with col_sb2:
    st.metric("Total Mass", f"{total_comp:.2f}%")

if not is_valid_composition:
    st.sidebar.markdown(
        f'<div style="font-size: 0.8rem; color: #d9755b; margin-top: 0.4rem;">'
        f'Solute mass ({sum_other:.2f}%) exceeds 100.00%. Titanium balance cannot be negative.'
        f'</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown('<div class="hairline-rule" style="margin: 1.2rem 0;"></div>', unsafe_allow_html=True)
sidebar_predict = st.sidebar.button("Calculate Young's Modulus", key="sidebar_calc_btn", use_container_width=True)

# ----------------- Feature Vector Construction -----------------
raw_composition = {
    column: (ti if column == 'Ti_wt' else alloy_elements.get(column[:-3], 0.0))
    for column in COMPOSITION_COLUMNS
}
raw_composition_data = pd.DataFrame([raw_composition])
raw_input_total = float(raw_composition_data.sum(axis=1).iloc[0])
input_data, input_ti_at, unknown_elements = prepare_model_input(
    raw_composition_data,
    FEATURES,
    COMPOSITION_COLUMNS,
)
# The saved pipeline one-hot encodes raw phase_est internally; keep its JSON schema.
ti_at_value = float(input_ti_at.iloc[0])
mo_eq_calc = float(input_data['Mo_eq'].iloc[0])
vec_calc = float(input_data['VEC'].iloc[0])
if abs(raw_input_total - 100.0) > 2.0:
    st.warning(
        f"Input composition totals {raw_input_total:.2f} wt%; it will be normalized to 100 wt%."
    )
if unknown_elements:
    st.warning(
        "Element(s) not present in the training columns will be ignored: "
        + ", ".join(unknown_elements)
    )
if ti_at_value < 60.0:
    st.warning(
        f"Ti_at is {ti_at_value:.2f} at%, outside the training range (minimum 60 at%)."
    )

# ----------------- Main Asymmetrical Layout (1.2 : 1.0) -----------------
col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.markdown('<div class="card-label">Composition Architecture</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 1.55rem; margin-top: 0;">Elemental Mass Fractions</h3>', unsafe_allow_html=True)
    
    # Active element breakdown
    all_elements = {'Ti (Matrix)': ti}
    for el_name, el_val in alloy_elements.items():
        if el_val > 0:
            all_elements[el_name] = el_val
            
    summary_rows = []
    for elem, val in all_elements.items():
        summary_rows.append({
            'Constituent': elem,
            'Mass Fraction': f"{val:.2f}%",
            'Raw': val
        })
    summary_df = pd.DataFrame(summary_rows)
    
    # Append clear total
    total_entry = pd.DataFrame([{
        'Constituent': 'Total Mass Fraction',
        'Mass Fraction': f"{total_comp:.2f}%",
        'Raw': total_comp
    }])
    render_df = pd.concat([summary_df, total_entry], ignore_index=True)
    
    st.dataframe(
        render_df[['Constituent', 'Mass Fraction']],
        use_container_width=True,
        hide_index=True
    )
    
    # Visual Fraction Representation
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Solute Distribution</div>', unsafe_allow_html=True)
    
    for elem, val in all_elements.items():
        pct = min(100.0, max(0.0, val))
        st.markdown(
            f'<div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.25rem;">'
            f'<span style="color: #c9c5bc;">{elem}</span>'
            f'<span class="mono-figure" style="color: #928b7e;">{val:.2f}%</span>'
            f'</div>'
            f'<div style="background: rgba(220, 215, 205, 0.08); height: 4px; border-radius: 1px; margin-bottom: 0.75rem;">'
            f'<div style="background: #c46849; width: {pct}%; height: 100%; border-radius: 1px;"></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Physical Descriptors Block
    st.markdown('<div class="hairline-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Phase Equilibrium Descriptors</div>', unsafe_allow_html=True)
    
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            f'<div style="font-size: 0.75rem; text-transform: uppercase; color: #8e887d; letter-spacing: 0.06em;">[Mo]eq Stability</div>'
            f'<div class="mono-figure" style="font-size: 1.4rem; color: #f2eee6; margin-top: 0.2rem;">{mo_eq_calc:.2f}</div>'
            f'<div style="font-size: 0.72rem; color: #787266; margin-top: 0.15rem;">Beta stabilizer potency</div>',
            unsafe_allow_html=True
        )
    with p2:
        st.markdown(
            f'<div style="font-size: 0.75rem; text-transform: uppercase; color: #8e887d; letter-spacing: 0.06em;">Valence (VEC)</div>'
            f'<div class="mono-figure" style="font-size: 1.4rem; color: #f2eee6; margin-top: 0.2rem;">{vec_calc:.2f}</div>'
            f'<div style="font-size: 0.72rem; color: #787266; margin-top: 0.15rem;">e⁻ per atom density</div>',
            unsafe_allow_html=True
        )
    with p3:
        st.markdown(
            f'<div style="font-size: 0.75rem; text-transform: uppercase; color: #8e887d; letter-spacing: 0.06em;">Beta Solutes</div>'
            f'<div class="mono-figure" style="font-size: 1.4rem; color: #f2eee6; margin-top: 0.2rem;">{beta_stabilizers:.2f}%</div>'
            f'<div style="font-size: 0.72rem; color: #787266; margin-top: 0.15rem;">Nb + Zr + Fe + Mo</div>',
            unsafe_allow_html=True
        )

with col_right:
    st.markdown('<div class="card-label">Computational Synthesis</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 1.55rem; margin-top: 0;">Evaluation & Validation</h3>', unsafe_allow_html=True)
    
    main_predict = st.button("Calculate Young's Modulus", key="main_calc_btn", use_container_width=True)
    
    if sidebar_predict or main_predict:
        if not is_valid_composition:
            st.markdown(
                '<div style="margin-top: 1.25rem; padding: 1rem; border-left: 3px solid #d9755b; background: rgba(217, 117, 91, 0.08); font-size: 0.85rem; color: #d9755b;">'
                'Cannot evaluate: Total solute mass exceeds 100.00%. Please rebalance elemental inputs.'
                '</div>',
                unsafe_allow_html=True
            )
            st.session_state['prediction'] = None
        else:
            try:
                model = load_model()
                pred = float(model.predict(input_data)[0])
                
                uncert_calc = 10.0
                conf_level = "Outside training range" if ti_at_value < 60.0 else "Moderate"
                
                st.session_state['prediction'] = pred
                st.session_state['predicted_ti'] = ti
                st.session_state['predicted_ti_at'] = ti_at_value
                st.session_state['predicted_mo_eq'] = mo_eq_calc
                st.session_state['predicted_vec'] = vec_calc
                st.session_state['uncertainty'] = uncert_calc
                st.session_state['confidence'] = conf_level
                
                formula_elements = [f"Ti-{ti:.1f}"]
                for el, v_val in alloy_elements.items():
                    if v_val > 0:
                        formula_elements.append(f"{v_val:.1f}{el}")
                st.session_state['predicted_alloy'] = "-".join(formula_elements)
                
            except Exception as e:
                st.markdown(
                    f'<div style="margin-top: 1.25rem; padding: 1rem; border-left: 3px solid #d9755b; background: rgba(217, 117, 91, 0.08); font-size: 0.85rem; color: #d9755b;">'
                    f'Model pipeline error: {e}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # If prediction exists, render the structured box, performance, and validation form
    if st.session_state['prediction'] is not None:
        pred_val = st.session_state['prediction']
        uncert = st.session_state.get('uncertainty', 10.0)
        conf = st.session_state.get('confidence', 'Moderate')
        nominal_formula = st.session_state.get('predicted_alloy', 'Ti-Alloy')
        
        # 1. Young's Modulus Prediction Box
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-box-header">Young's Modulus Prediction</div>
                <div class="prediction-box-value mono-figure">{pred_val:.2f} <span class="prediction-box-unit">GPa</span></div>
                <div class="prediction-box-meta">
                    <div>Estimated uncertainty: <span class="mono-figure" style="font-weight: 500; color: #f0ece3;">±{uncert:.2f} GPa</span></div>
                    <div>expected error about +/-10 GPa</div>
                    <div>Confidence: <span class="mono-figure" style="font-weight: 500; color: #f0ece3;">{conf}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 2. Model Performance Block — from actual saved metrics
        cv_r2 = metrics.get("cv_r2_mean", float('nan'))
        cv_mae = metrics.get("cv_mae_gpa", float('nan'))
        cv_rmse = metrics.get("cv_rmse_gpa", float('nan'))
        test_r2 = metrics.get("test_r2", float('nan'))
        test_mae = metrics.get("test_mae_gpa", float('nan'))
        test_rmse = metrics.get("test_rmse_gpa", float('nan'))
        model_name = metrics.get("selected_model", "Unknown")
        n_train = metrics.get("training_rows", "?")
        n_test = metrics.get("test_rows", "?")

        st.markdown(
            f"""
            <div class="bench-section">
                <div class="bench-title">Model Performance — {model_name}</div>
                <div class="bench-divider"></div>
                <div class="bench-row"><span class="bench-row-label">Training Samples</span><span class="bench-row-value">{n_train}</span></div>
                <div class="bench-row"><span class="bench-row-label">Test Samples</span><span class="bench-row-value">{n_test}</span></div>
                <div class="bench-divider"></div>
                <div class="bench-row"><span class="bench-row-label">5-Fold CV R²</span><span class="bench-row-value">{cv_r2:.2f}</span></div>
                <div class="bench-row"><span class="bench-row-label">5-Fold CV MAE</span><span class="bench-row-value">{cv_mae:.2f} GPa</span></div>
                <div class="bench-row"><span class="bench-row-label">5-Fold CV RMSE</span><span class="bench-row-value">{cv_rmse:.2f} GPa</span></div>
                <div class="bench-divider"></div>
                <div class="bench-row"><span class="bench-row-label">Test R²</span><span class="bench-row-value">{test_r2:.2f}</span></div>
                <div class="bench-row"><span class="bench-row-label">Test MAE</span><span class="bench-row-value">{test_mae:.2f} GPa</span></div>
                <div class="bench-row"><span class="bench-row-label">Test RMSE</span><span class="bench-row-value">{test_rmse:.2f} GPa</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 3. External Validation Block (Interactive Form Type)
        st.markdown(
            f"""
            <div class="bench-section" style="margin-bottom: 0.5rem;">
                <div class="bench-title">External Validation</div>
                <div class="bench-divider"></div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #c46849; margin-bottom: 0.6rem; font-weight: 600;">
                    {nominal_formula}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Form for entering experimental value
        with st.form("external_validation_form"):
            st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #928b7e; margin-bottom: 0.3rem;">Experimental Benchmark Input</div>', unsafe_allow_html=True)
            user_input = st.number_input(
                "Enter experimental value (GPa):",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state['exp_val']) if st.session_state['exp_val'] is not None else 0.0,
                step=0.5,
                format="%.2f",
                help="Type the measured laboratory Young's Modulus in GPa"
            )
            calc_error_btn = st.form_submit_button("Record & Calculate Validation Error", use_container_width=True)
            
            if calc_error_btn:
                if user_input > 0.0:
                    st.session_state['exp_val'] = float(user_input)
                else:
                    st.session_state['exp_val'] = None

        # Display Validation Metrics
        curr_exp = st.session_state.get('exp_val', None)
        if curr_exp is not None and curr_exp > 0.0:
            err_pct = abs(pred_val - curr_exp) / curr_exp * 100.0
            exp_display = f"{curr_exp:.2f} GPa"
            err_display = f"{err_pct:.2f}%"
        else:
            exp_display = '<span style="color: #9c9586; font-style: italic;">enter value above</span>'
            err_display = '<span style="color: #9c9586;">--</span>'
            
        st.markdown(
            f"""
            <div style="background: #17181a; border: 1px solid rgba(220, 215, 205, 0.12); border-radius: 3px; padding: 0.9rem 1.1rem; margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.2rem 0;"><span style="color: #928b7e;">Experimental:</span><span style="color: #f0ece3;">{exp_display}</span></div>
                <div style="display: flex; justify-content: space-between; padding: 0.2rem 0;"><span style="color: #928b7e;">Predicted:</span><span style="color: #f0ece3; font-weight: 500;">{pred_val:.2f} GPa</span></div>
                <div style="display: flex; justify-content: space-between; padding: 0.2rem 0; border-top: 1px dashed rgba(220, 215, 205, 0.15); margin-top: 0.3rem; padding-top: 0.4rem;"><span style="color: #928b7e;">Error:</span><span style="color: #c46849; font-weight: 600;">{err_display}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Caution Note
        st.markdown(
            """
            <div class="caution-note">
                <strong>Notice:</strong> Prediction depends on composition, phase distribution, and thermo-mechanical processing conditions.
            </div>
            """,
            unsafe_allow_html=True
        )
        
    else:
        st.markdown(
            '<div style="margin-top: 1.25rem; padding: 1.8rem; border: 1px dashed rgba(220, 215, 205, 0.18); border-radius: 2px; text-align: left;">'
            '<div class="card-label">Evaluation Pending</div>'
            '<div style="font-size: 0.88rem; color: #9c9586; line-height: 1.6; margin-top: 0.5rem;">'
            'Configure elemental mass fractions in the specification panel on the left, then select '
            "<strong>Calculate Young's Modulus</strong> to evaluate the formulation, view model uncertainty, and benchmark against external experimental data."
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

# ----------------- Editorial Laboratory Footer -----------------
st.markdown(
    '<div class="editorial-footer">'
    '<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">'
    '<div>'
    '<strong>Materials Informatics Laboratory</strong> · Computational Metallurgy Division<br>'
    'Specification Reference: TSTM-2691685 · Empirical Phase Calibration Dataset'
    '</div>'
    '<div style="text-align: right;" class="mono-figure">'
    'VALENCE ENSEMBLE ENGINE v2.4<br>'
    'SYSTEM STATUS: OPERATIONAL'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)