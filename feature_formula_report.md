# Feature Formula and Definition Report

## Purpose and calculation sequence

The model predicts `E_GPa` from alloy composition. Composition columns ending in
`_wt` are the element weight percentages. For feature calculations, the
composition is normalized to 100 wt%, then converted to atomic fractions when a
descriptor is atom-based. For element \(i\), with weight percentage \(w_i\) and
atomic mass \(M_i\):

\[
x_i = \frac{w_i/M_i}{\sum_j w_j/M_j}
\]

Here, \(x_i\) is the atomic fraction and \(\sum_i x_i=1\). Weight-based
equivalents use the normalized weight percentages directly. Element constants
such as valence, radius, and Morinaga parameters are looked up in the tables in
[`features.py`](./features.py).

## Model feature columns

| Feature column | Calculation in this project | One-line definition |
|---|---|---|
| `*_wt` (for example `Ti_wt`, `Nb_wt`) | Entered element weight percentages, normalized so the composition totals 100 wt%. | The mass fraction of an element in the alloy. |
| `VEC` | \(\displaystyle \sum_i x_i\,\mathrm{VEC}_i\), using the element valence values in `VEC_TABLE`. | The composition-weighted average number of valence electrons per atom. |
| `delta_r` | \(\displaystyle 100\sqrt{\sum_i x_i\left(1-\frac{r_i}{\bar r}\right)^2}\), where \(\bar r=\sum_i x_i r_i\), and \(r_i\) is the metallic radius in pm. | The atomic-size mismatch among alloying elements, expressed as a percentage. |
| `delta_S_mix` | \(\displaystyle -R\sum_i x_i\ln(x_i)\), with \(R=8.314462618\ \mathrm{J\,mol^{-1}K^{-1}}\); terms with \(x_i=0\) contribute zero. | The ideal configurational mixing entropy based on atomic fractions. |
| `Mo_eq` | \(\displaystyle \mathrm{Mo}+0.2\mathrm{Ta}+0.28\mathrm{Nb}+0.4\mathrm{W}+0.67\mathrm{V}+1.25\mathrm{Cr}+1.25\mathrm{Ni}+1.7\mathrm{Mn}+1.7\mathrm{Co}+2.5\mathrm{Fe}-\mathrm{Al}\), using wt% values. | A weighted estimate of beta-stabilizing strength expressed as Mo-equivalent wt%. |
| `Al_eq` | \(\displaystyle \mathrm{Al}+\frac{\mathrm{Zr}}{6}+\frac{\mathrm{Sn}}{3}+10\mathrm{O}\), using wt% values. | A weighted estimate of alpha-stabilizing strength expressed as Al-equivalent wt%. |
| `Bo` | \(\displaystyle \sum_i x_i\,\mathrm{Bo}_i\), using the first value in each element's `MORINAGA` table pair. | The atomic-fraction-weighted Morinaga bond-order descriptor. |
| `Md` | \(\displaystyle \sum_i x_i\,\mathrm{Md}_i\), using the second value in each element's `MORINAGA` table pair. | The atomic-fraction-weighted Morinaga d-orbital energy descriptor. |
| `phase_est` | A rule-based label described below; the model's preprocessing one-hot encodes the label into indicator columns. | A coarse composition-based screening category for likely alpha, alpha+beta, or beta phase stability. |

### Phase screening rules

`phase_est` is a heuristic label derived from composition, not a measured phase
fraction or a substitute for heat-treatment and microstructure data. The
implementation applies these rules in order:

1. If composition is invalid, classify as `uncertain`.
2. If `Ti_wt < 70`, classify as `uncertain`.
3. If `Al_eq >= 8 wt%`, classify as `uncertain`.
4. Otherwise, if `Mo_eq < 2.5 wt%`, classify as `alpha`.
5. Otherwise, if `2.5 <= Mo_eq <= 10 wt%`, classify as `alpha+beta`.
6. If `Mo_eq > 10 wt%`, classify as `beta`.

The rules use the thresholds declared in `features.py`. The `phase_note` and
`ti_based` columns are also generated for explanation/filtering, but are not
included in the saved v2 model feature list.

## Missing values and model input

Feature calculations require a positive-total, nonnegative composition. When
an element with a nonzero amount lacks a required lookup value, the affected
descriptor is left missing (`NaN`) rather than assigning a made-up constant.
For example, missing valence data affects `VEC`; missing radius data affects
`delta_r`; and missing Morinaga data affects both `Bo` and `Md`.

The model uses the 19 composition columns, the seven numerical descriptors
(`VEC`, `delta_r`, `delta_S_mix`, `Mo_eq`, `Al_eq`, `Bo`, `Md`), and
`phase_est`. It does not use `E_GPa` as an input. In the app, the saved model
pipeline handles one-hot encoding of `phase_est`.

## Implementation reference

The equations, atomic masses, valence values, metallic radii, Mo-equivalent
coefficients, Al-equivalent coefficients, Morinaga values, and phase thresholds
are implemented in [`features.py`](./features.py). This report describes that
implementation; it does not introduce additional descriptors or constants.
