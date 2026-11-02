from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "final_report.docx"


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.autofit = True
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell_text(cell, text, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            set_cell_text(cell, text)
    document.add_paragraph()


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(end)


def body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.line_spacing = 1.08
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.first_line_indent = Inches(0.25)


def heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True


def add_figure(document: Document, filename: str, width: float, caption: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(ROOT / filename), width=Inches(width))
    add_caption(document, caption)


def main() -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for style_name in ["Title", "Heading 1", "Heading 2"]:
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    document.styles["Title"].font.size = Pt(16)
    document.styles["Heading 1"].font.size = Pt(12)
    document.styles["Heading 1"].font.bold = True
    document.styles["Heading 2"].font.size = Pt(11)
    document.styles["Heading 2"].font.bold = True
    add_page_number(section.footer.paragraphs[0])

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(30)
    title.paragraph_format.space_after = Pt(18)
    run = title.add_run(
        "Machine-Learning Prediction of Young's Modulus of Titanium Alloys "
        "from Chemical Composition"
    )
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)
    for line in [
        "<TODO: author name>",
        "<TODO: department, college>",
        "Supervisor: <TODO: professor name>",
        "Date: <TODO: date>",
    ]:
        p = document.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph()
    heading(document, "Abstract", 1)
    body(
        document,
        "Low Young's modulus in beta-titanium alloys is desirable for biomedical "
        "implants because it can reduce stress shielding. Experimental modulus "
        "measurement is slow, while literature data is scattered across sources. "
        "This study evaluated prediction of Young's modulus, E, in GPa from "
        "composition in weight percent using elemental and derived descriptors. "
        "Fifteen CSV files were merged; computed rows were removed; Ti-matrix "
        "compositions were selected; and repeated compositions were collapsed by "
        "median modulus before grouped splitting. A composition-plus-engineered "
        "feature XGBoost regressor with an absolute-error objective and default "
        "parameters was selected by five-fold grouped cross-validation. On the "
        "held-out test set of 151 rows, MAE was 9.88 GPa, RMSE was 13.00 GPa, and "
        "R2 was 0.684. The test MAE was about 1.12 times the estimated 8.78 GPa "
        "within-composition noise floor. Engineered features did not clearly "
        "improve grouped-CV error over composition alone. The results supported "
        "composition-based screening but did not establish processing-independent "
        "property prediction."
    )

    document.add_page_break()
    heading(document, "1 Introduction")
    body(
        document,
        "Titanium alloys are used in biomedical implants, where a lower elastic "
        "modulus can reduce stiffness mismatch with bone and thereby reduce stress "
        "shielding. Beta-titanium alloys are of particular interest. Young's modulus "
        "is measured experimentally, but measurement is slow and reported values "
        "are dispersed across the literature. A composition-based predictor could "
        "help prioritize candidate compositions for measurement; it cannot replace "
        "experiments that account for processing and phase constitution. "
        "[reference needed]"
    )
    body(
        document,
        "The objective was to predict E in GPa from alloy composition in wt%. "
        "Elemental composition alone was compared with composition plus engineered "
        "descriptors. Model selection used only the training partition and "
        "composition-grouped folds; the held-out test partition was evaluated once."
    )
    heading(document, "2 Data")
    body(
        document,
        "The initial merged table contained 897 rows from 15 CSV files, comprising "
        "literature compilations, earlier project datasets, values digitized from "
        "published figures, and density-functional-theory (DFT) values from "
        "Materials Project and two DFT figures. Figure-digitized values had an "
        "estimated reading error of approximately +/-0.5 to 2 GPa. Compositions "
        "were represented in wt% and the target was E in GPa. at% compositions "
        "were converted to wt% with standard atomic masses, and each row was "
        "normalized to 100 wt%."
    )
    body(
        document,
        "Computed rows were excluded because they sat systematically apart from "
        "experimental values: 38 matched the Materials Project file and 11 matched "
        "two DFT figures, for 49 removals. Of the 848 remaining rows, 31 with "
        "atomic Ti below 60 at% were removed, leaving 817 Ti-matrix rows. This "
        "atomic-percent criterion retained heavy-element beta alloys that a wt% "
        "Ti cutoff would discard; Ti-29Nb-13Ta-4.6Zr had 53.4 wt% Ti but about "
        "72 at% Ti."
    )
    body(
        document,
        "Compositions equal after rounding all element wt% values to one decimal "
        "place were collapsed to one record with median E. The data were reduced "
        "from 817 rows to 754 composition groups. Thirty groups had repeated "
        "values; their mean within-group sample standard deviation was 8.78 GPa. "
        "This was treated as an empirical label-noise estimate, not a strict "
        "lower bound on prediction error."
    )
    body(
        document,
        "An 80/20 GroupShuffleSplit with seed 42 allocated 603 rows to training "
        "and 151 to testing, with no rounded composition shared across partitions. "
        "The test partition included 46 rows with Ti below 70 wt%; this was a "
        "reported subgroup, not the inclusion criterion."
    )
    add_table(
        document,
        ["Partition", "n", "Mean E", "SD", "Min", "Max (GPa)"],
        [
            ["Train", "603", "73.1", "19.8", "23.0", "157.0"],
            ["Test", "151", "75.7", "23.2", "29.5", "149.0"],
        ],
    )
    add_caption(document, "Table 1. Young's modulus distribution by split (GPa).")

    document.add_page_break()
    heading(document, "3 Methods")
    heading(document, "3.1 Composition-derived descriptors", 2)
    body(
        document,
        "Feature set A comprised all element wt% columns. Feature set B comprised "
        "A plus VEC, atomic-size mismatch delta_r, ideal configurational mixing "
        "entropy delta_S_mix, Mo_eq, Al_eq, Morinaga Bo and Md, and the categorical "
        "phase_est screening label. The label used Mo_eq thresholds of 2.5 and "
        "10 wt%; it was marked uncertain when Al_eq was at least 8 wt% or the "
        "alloy was Ti-poor (the operational threshold was not specified in the "
        "supplied report facts). The Appendix gives the formulas. Atomic descriptors "
        "used atomic fractions obtained from normalized wt% composition and "
        "standard atomic masses. The saved inference pipeline handled phase_est "
        "as a categorical input."
    )
    heading(document, "3.2 Models and validation", 2)
    body(
        document,
        "The evaluated models were XGBoost regressors with squared-error and "
        "absolute-error objectives, Random Forest, Ridge, and a mean-predictor "
        "baseline. Ridge results were omitted because they were unavailable in "
        "the v2 report. Five-fold GroupKFold used the training data only, with "
        "rounded composition groups. MAE was the primary selection metric; RMSE "
        "and R2 were also recorded. RandomizedSearchCV used 60 iterations for "
        "absolute-error XGBoost on sets A and B. The held-out test set was used "
        "only after selection."
    )
    body(
        document,
        "The final model was the XGBoost regressor with feature set B, objective "
        "reg:absoluteerror, and default parameters. Software versions were pandas "
        "3.1.0, NumPy 2.5.3, scikit-learn 1.9.1, and XGBoost 3.4.1. The random "
        "seed was 42."
    )

    document.add_page_break()
    heading(document, "4 Results")
    heading(document, "4.1 Cross-validation and model selection", 2)
    body(
        document,
        "The default set B XGBoost model with the absolute-error objective had "
        "five-fold grouped-CV MAE 10.19 +/- 0.63 GPa. Random Forest obtained "
        "10.40 +/- 0.97 GPa for A and 10.21 +/- 1.14 GPa for B. The mean baseline "
        "obtained 15.39 +/- 1.24 GPa for both sets. Tuned absolute-error XGBoost "
        "obtained 10.59 +/- 0.77 GPa for A and 10.43 +/- 0.91 GPa for B; tuning "
        "did not improve on the selected default B model."
    )
    add_table(
        document,
        ["Set", "Model", "CV MAE (mean +/- SD), GPa"],
        [
            ["A", "XGBoost, absolute error", "11.10 +/- 0.46"],
            ["A", "XGBoost, squared error", "11.20 +/- 0.29"],
            ["A", "Random Forest", "10.40 +/- 0.97"],
            ["A", "Mean baseline", "15.39 +/- 1.24"],
            ["B", "XGBoost, absolute error", "10.19 +/- 0.63"],
            ["B", "XGBoost, squared error", "10.79 +/- 0.92"],
            ["B", "Random Forest", "10.21 +/- 1.14"],
            ["B", "Mean baseline", "15.39 +/- 1.24"],
            ["A", "Tuned XGBoost, absolute error", "10.59 +/- 0.77"],
            ["B", "Tuned XGBoost, absolute error", "10.43 +/- 0.91"],
        ],
    )
    add_caption(document, "Table 2. Five-fold grouped-CV MAE on the training partition.")
    body(
        document,
        "The selected model's test MAE was 9.88 GPa, RMSE was 13.00 GPa, and "
        "R2 was 0.684 (n=151). Subgroup MAE was 7.78 GPa for 28 rows labelled "
        "beta by the screening rule, 11.51 GPa for 41 rows with E below 60 GPa, "
        "and 7.80 GPa for 46 rows with Ti below 70 wt%."
    )
    add_figure(
        document,
        "parity_test.png",
        4.7,
        "Figure 1. Held-out test parity plot; the diagonal indicates perfect agreement.",
    )

    document.add_page_break()
    heading(document, "4 Results (continued)")
    heading(document, "4.2 Composition-series plots and feature comparison", 2)
    body(
        document,
        "The mean paired MAE gain from B over A was 0.17 GPa, smaller than one "
        "fold standard deviation for tuned B (0.91 GPa). The engineered descriptors "
        "therefore did not clearly improve performance over composition alone."
    )
    body(
        document,
        "Figure 2 compares predicted curves with digitized Ti-Nb-xZr data at "
        "x=0, 4, 8, 12, and 18 at% Zr. The requested overlap check found 28 of "
        "44 points matching training rows within 0.5 wt% composition and 0.5 GPa "
        "in E. The comparison therefore includes training observations and is "
        "a partly in-sample fit visualization, not independent validation. "
        "test_report.txt was not present; the count came from the explicit "
        "comparison against train_v2.csv."
    )
    add_figure(
        document,
        "nb_zr_vs_real.png",
        6.1,
        "Figure 2. Predicted Ti-Nb-xZr curves and digitized points at five Zr at% levels; 28/44 points overlap training rows at the stated tolerances.",
    )
    add_figure(
        document,
        "nb_variation_graph.png",
        5.3,
        "Figure 3. Predicted modulus over the binary Ti-Nb wt% sweep; this is model output, not a new experiment.",
    )

    document.add_page_break()
    heading(document, "5 Discussion")
    body(
        document,
        "The test MAE of 9.88 GPa was about 1.12 times the estimated label-noise "
        "floor of 8.78 GPa. The error scale was therefore similar to the "
        "within-composition variability observed in duplicate groups. The floor "
        "was estimated from only 30 groups with repeated values and was not a "
        "universal lower bound."
    )
    body(
        document,
        "The engineered descriptors gave a small average improvement over "
        "composition alone, but it was smaller than fold-to-fold variability. "
        "The earlier model, before DFT exclusion, Ti-matrix filtering, and "
        "duplicate collapse, had test MAE 11.53 GPa and R2 0.52. Since data "
        "preparation and splitting changed, this comparison did not isolate the "
        "causal effect of any individual cleaning step."
    )
    body(
        document,
        "The largest test errors were reported for dilute alloys with observed "
        "moduli around 91-113 GPa that were predicted around 66-79 GPa. Examples "
        "were Ti-3Mo-1Nb, Ti-3Mo-1Al, Ti-7.4Cr, and Ti-8Mo-1Nb. Composition did "
        "not encode processing history or retained phase. Similar chemistries "
        "after different processing, including quenched beta and annealed "
        "alpha+beta conditions, could therefore have different modulus values "
        "that the input features could not distinguish. [reference needed]"
    )
    heading(document, "6 Limitations")
    body(
        document,
        "The dataset mixed tensile, ultrasonic, nanoindentation, and compression "
        "measurements with values digitized from figures. Digitization carried "
        "estimated reading errors of +/-0.5 to 2 GPa, and processing histories "
        "and measurement protocols were not harmonized. No processing, heat "
        "treatment, or measured phase variable was included."
    )
    body(
        document,
        "The phase label was a coarse composition-equivalent screen, not a "
        "measured phase state or thermodynamic calculation. Tree models produce "
        "stepwise responses and may not reproduce sharp phase-related dips, "
        "including the reported 38.5 GPa value at Ti-14Nb-18Zr. The low-modulus "
        "test subset had only 41 rows, so its MAE was noisy. Extrapolation beyond "
        "the training composition range was unreliable. Predictions should be "
        "treated as approximate, with expected error about +/-10 GPa, not as "
        "substitutes for measurement."
    )

    document.add_page_break()
    heading(document, "7 Conclusion")
    body(
        document,
        "Grouped evaluation produced test MAE 9.88 GPa for the selected default "
        "XGBoost model using composition and engineered descriptors; test R2 "
        "was 0.684. Engineered features did not clearly outperform composition "
        "alone by more than fold variability. Prediction error was similar in "
        "scale to the estimated within-composition standard deviation, while "
        "process and phase information remained important limitations."
    )
    body(
        document,
        "Future work should add processing and phase columns, source and "
        "measurement-method labels, more low-modulus beta-alloy experiments, "
        "and uncertainty estimates."
    )
    heading(document, "References")
    body(
        document,
        "[reference needed: no bibliographic entries were supplied. Add the actual "
        "data-source papers and any standard references used before submission.]"
    )
    heading(document, "Appendix: Feature formulas and tables")
    body(
        document,
        "Let x_i be the atomic fraction of element i, with sum_i x_i=1, and r_i "
        "its metallic radius. Let r_mean=sum_i x_i r_i. The composition was "
        "normalized to 100 wt% and converted to atomic fractions using standard "
        "atomic masses. The formulas below were used."
    )
    for equation in [
        "VEC = sum_i (x_i * VEC_i)",
        "delta_r (%) = 100 * sqrt(sum_i [x_i * (1 - r_i/r_mean)^2])",
        "delta_S_mix = -R * sum_i (x_i * ln(x_i)); zero terms omitted",
        "Mo_eq (wt%) = Mo + 0.2Ta + 0.28Nb + 0.4W + 0.67V + 1.25Cr + 1.25Ni + 1.7Mn + 1.7Co + 2.5Fe - Al",
        "Al_eq (wt%) = Al + Zr/6 + Sn/3 + 10O",
        "Bo = sum_i (x_i * Bo_i); Md = sum_i (x_i * Md_i)",
    ]:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(equation)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)

    add_table(
        document,
        ["Feature group", "Definition / interpretation"],
        [
            ["A", "All element composition columns in wt%."],
            [
                "B",
                "Set A plus VEC, delta_r, delta_S_mix, Mo_eq, Al_eq, Bo, Md, and phase_est.",
            ],
            [
                "phase_est",
                "Mo_eq < 2.5 wt%: alpha; 2.5-10 wt% inclusive: alpha+beta; >10 wt%: beta. Uncertain for Al_eq >= 8 wt% or Ti-poor alloy (threshold not specified).",
            ],
        ],
    )
    add_caption(document, "Table 3. Model feature groups and phase-screen interpretation.")
    body(
        document,
        "Mo_eq and Al_eq used wt% composition; VEC, delta_r, Bo, and Md used "
        "atomic fractions. Feature definitions and tabulated element constants "
        "require citations before external publication. [reference needed]"
    )
    body(
        document,
        "Numerical provenance: reported quantities were taken from the supplied "
        "task facts; the 28/44 overlap count was calculated by comparing the "
        "supplied real-point list with train_v2.csv using the specified tolerances. "
        "See number_source_audit.txt for the value-to-prompt mapping. Author, "
        "affiliation, supervisor, date, and bibliography remain TODO items."
    )

    document.save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
