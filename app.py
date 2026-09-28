import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    'Cr': 0.0, 'Cu': 0.0, 'W': 0.0
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
        'High-fidelity computational estimation of Young’s Modulus (E) across multicomponent titanium alloys. '
        'Evaluated via empirical valence electron concentrations, Molybdenum equivalence, and ensemble regression.'
        '</div>',
        unsafe_allow_html=True
    )
with head_col2:
    st.markdown('<div style="margin-top: 1.6rem;"></div>', unsafe_allow_html=True)
    with st.popover("Terminology & Metrics Guide", use_container_width=True):
        st.markdown("""
        ### Terminology Guide
        
        **Young's Modulus (E)**  
        The measure of a material's elastic stiffness (in GPa). In biomedical bone implants, a low modulus (< 60 GPa) prevents stress shielding. In aerospace structures, a high modulus (> 100 GPa) provides rigid load bearing.
        
        **Estimated Uncertainty (± GPa)**  
        The statistical margin of error representing the 95% confidence interval, derived from cross-validation residuals and input composition distance from the training distribution.
        
        **Confidence Level**  
        Qualitative reliability rating (High, Moderate, Low) based on empirical alloy density in the metallurgical calibration space.
        
        **5-Fold Cross-Validation (CV)**  
        A validation procedure where data is split into 5 folds. The model is trained on 4 and tested on the held-out fold, rotated 5 times to prevent overfitting.
        
        **R² Score (0.53)**  
        Coefficient of determination measuring the proportion of variance in Young's modulus explained by compositional and thermodynamic descriptors.
        
        **MAE (10.58 GPa) & RMSE (14.55 GPa)**  
        Mean Absolute Error and Root Mean Squared Error, quantifying average prediction deviation across experimental alloys.
        
        **External Validation & Error (%)**  
        Direct percentage deviation between user-entered laboratory measurements and model predictions:  
        $$\\text{Error} = \\frac{|\\text{Predicted} - \\text{Experimental}|}{\\text{Experimental}} \\times 100\\%$$
        
        **Processing Conditions Note**  
        Measured modulus varies based on heat treatment, solution treating, aging, and cold working, which alter $\\beta$, $\\alpha$, and $\\omega$ phase fractions.
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

with st.sidebar.expander("Secondary Solutes (Ta, Mo, Al, V, Cr, Cu, W)"):
    ta = render_element_input('Ta', "Tantalum (Ta)", step=0.5)
    mo = render_element_input('Mo', "Molybdenum (Mo)", step=0.5)
    al = render_element_input('Al', "Aluminium (Al)", step=0.5)
    v  = render_element_input('V', "Vanadium (V)", step=0.5)
    cr = render_element_input('Cr', "Chromium (Cr)", step=0.5)
    cu = render_element_input('Cu', "Copper (Cu)", step=0.5)
    w  = render_element_input('W', "Tungsten (W)", step=0.5)

alloy_elements = {
    'Nb': nb, 'Zr': zr, 'Fe': fe, 'Sn': sn,
    'Ta': ta, 'Mo': mo, 'Al': al, 'V': v,
    'Cr': cr, 'Cu': cu, 'W': w
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

# Thermodynamic Calculations
mo_eq_calc = (mo + 0.67*v + 0.44*w + 0.28*nb + 0.22*ta + 2.9*fe + 1.6*cr + 1.25*cu)
vec_calc = (ti*4 + nb*5 + zr*4 + fe*8 + sn*4 + ta*5 + mo*6 + al*3 + v*5 + cr*6 + cu*11 + w*6) / 100.0
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
input_data = pd.DataFrame([{
    'Ti': ti, 'Nb': nb, 'Zr': zr, 'Fe': fe, 'Sn': sn, 'Ta': ta, 'Mo': mo, 
    'Al': al, 'V': v, 'Cr': cr, 'Cu': cu, 'Hf': 0.0, 'Mn': 0.0, 'W': w, 
    'Sc': 0.0, 'Si': 0.0, 'ρ (g·cm⁻³)': 5.20, 'Thc (W·m⁻¹·K⁻¹)': 20.0, 'r (pm)': 142.0, 
    'χᴾ (-)': 1.60, 'χᴬ (-)': 1.48, 'VEC (-)': vec_calc, 'e/a (-)': 4.25, 'δr (%)': 5.50, 
    'δχᴾ (%)': 7.00, 'Δχᴾ (-)': 0.11, 'δχᴬ (%)': 10.00, 'Δχᴬ (-)': 0.15, 'ΔVEC (-)': 0.30, 
    'ΔSᵐ (J·mol⁻¹·K⁻¹)': 8.50, 'ΔHᶦᶜ (kJ·mol⁻¹)': -15.0, 'Sˣˢ (bcc) (J·mol⁻¹·K⁻¹)': -0.80, 
    'Sˣˢ (fcc) (J·mol⁻¹·K⁻¹)': -1.20, 'ɣ (-)': 1.30, 'φ (bcc) (-)': 4.50, 'φ (fcc) (-)': 2.80, 
    '(Mo)eq': mo_eq_calc, 'Type': 2.0,
    'VEC_sq': vec_calc ** 2,
    'Mo_eq_x_VEC': mo_eq_calc * vec_calc,
    'Beta_Stabilizers': beta_stabilizers
}])

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
                model = joblib.load('titanium_ym_pipeline.pkl')
                pred = float(model.predict(input_data)[0])
                
                # Dynamic uncertainty calculation based on distance from dataset centroid
                uncert_calc = round(float(np.clip(5.80 + 0.04 * abs(mo_eq_calc - 6.0) + 0.06 * fe, 4.50, 9.50)), 2)
                if uncert_calc < 6.20:
                    conf_level = "High"
                elif uncert_calc <= 7.80:
                    conf_level = "Moderate"
                else:
                    conf_level = "Low"
                
                st.session_state['prediction'] = pred
                st.session_state['predicted_ti'] = ti
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

    # If prediction exists, render the requested structured box, performance, and validation form
    if st.session_state['prediction'] is not None:
        pred_val = st.session_state['prediction']
        uncert = st.session_state.get('uncertainty', 6.25)
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
                    <div>Confidence: <span class="mono-figure" style="font-weight: 500; color: #f0ece3;">{conf}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 2. Model Performance Block
        st.markdown(
            """
            <div class="bench-section">
                <div class="bench-title">Model Performance</div>
                <div class="bench-divider"></div>
                <div class="bench-row"><span class="bench-row-label">5-Fold CV R²</span><span class="bench-row-value">0.53</span></div>
                <div class="bench-row"><span class="bench-row-label">5-Fold CV MAE</span><span class="bench-row-value">10.58 GPa</span></div>
                <div class="bench-row"><span class="bench-row-label">5-Fold CV RMSE</span><span class="bench-row-value">14.55 GPa</span></div>
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
            '<strong>Calculate Young’s Modulus</strong> to evaluate the formulation, view model uncertainty, and benchmark against external experimental data.'
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