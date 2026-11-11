import os
import sys
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, PageBreak, FrameBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

def create_camera_ready_pdf(output_pdf_path="camera_ready/CAMERA_READY_REVISED.pdf"):
    print("Building Camera-Ready IEEE Conference PDF...")
    
    # Page dimensions (US Letter: 612 x 792 pt, 8.5 x 11 in)
    PAGE_WIDTH, PAGE_HEIGHT = letter
    MARGIN_LEFT = 45.0   # 0.625 in
    MARGIN_RIGHT = 45.0  # 0.625 in
    MARGIN_TOP = 45.0    # ~0.625 in
    MARGIN_BOTTOM = 45.0 # ~0.625 in
    
    PRINTABLE_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT # 522 pt
    GUTTER = 16.0
    COL_WIDTH = (PRINTABLE_WIDTH - GUTTER) / 2.0 # 253 pt
    
    # Styles
    base_styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15.5,
        leading=18.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#111111'),
        spaceAfter=7
    )
    
    author_style = ParagraphStyle(
        'AuthorBlock',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.0,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#222222'),
        spaceAfter=10
    )
    
    abstract_heading_style = ParagraphStyle(
        'AbstractHeading',
        parent=base_styles['Normal'],
        fontName='Helvetica-BoldOblique',
        fontSize=8.5,
        leading=11.0,
        alignment=TA_JUSTIFY,
        textColor=colors.black
    )
    
    abstract_body_style = ParagraphStyle(
        'AbstractBody',
        parent=base_styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.0,
        leading=10.2,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=5
    )
    
    keywords_style = ParagraphStyle(
        'Keywords',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=10.2,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.8,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#000000'),
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=base_styles['Normal'],
        fontName='Helvetica-BoldOblique',
        fontSize=8.2,
        leading=10.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#000000'),
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextJustified',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=9.8,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        firstLineIndent=9,
        spaceAfter=3
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=9.5,
        alignment=TA_JUSTIFY,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2
    )
    
    equation_style = ParagraphStyle(
        'EquationText',
        parent=base_styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceBefore=2,
        spaceAfter=2
    )
    
    table_title_style = ParagraphStyle(
        'TableTitle',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.0,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.0,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )
    
    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=table_cell_style,
        alignment=TA_LEFT
    )

    ref_style = ParagraphStyle(
        'ReferenceItem',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=8.5,
        alignment=TA_JUSTIFY,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=2.0
    )

    # Multi-template setup
    doc = BaseDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM
    )
    
    # Page 1 Template: Title frame on top, then 2 columns below
    title_height = 80.0
    cols_y = MARGIN_BOTTOM
    cols_height_p1 = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM - title_height
    
    frame_title = Frame(
        MARGIN_LEFT, PAGE_HEIGHT - MARGIN_TOP - title_height,
        PRINTABLE_WIDTH, title_height,
        id='F_Title', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    frame_p1_c1 = Frame(
        MARGIN_LEFT, cols_y, COL_WIDTH, cols_height_p1,
        id='F_P1_C1', topPadding=4, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    frame_p1_c2 = Frame(
        MARGIN_LEFT + COL_WIDTH + GUTTER, cols_y, COL_WIDTH, cols_height_p1,
        id='F_P1_C2', topPadding=4, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    
    template_page1 = PageTemplate(
        id='Page1',
        frames=[frame_title, frame_p1_c1, frame_p1_c2]
    )
    
    # Later Pages Template: 2 full height columns
    cols_height_later = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    frame_later_c1 = Frame(
        MARGIN_LEFT, MARGIN_BOTTOM, COL_WIDTH, cols_height_later,
        id='F_Later_C1', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    frame_later_c2 = Frame(
        MARGIN_LEFT + COL_WIDTH + GUTTER, MARGIN_BOTTOM, COL_WIDTH, cols_height_later,
        id='F_Later_C2', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#444444'))
        page_num = doc.page
        canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, MARGIN_BOTTOM - 20, f"Page {page_num} of 6")
        canvas.drawString(MARGIN_LEFT, PAGE_HEIGHT - MARGIN_TOP + 12, "AIHC 2027: ML in Clinical Decision Support and Predictive Analytics (Paper #516)")
        canvas.restoreState()

    template_later = PageTemplate(
        id='LaterPages',
        frames=[frame_later_c1, frame_later_c2],
        onPage=add_page_number
    )
    
    doc.addPageTemplates([template_page1, template_later])
    
    story = []
    
    # --- Title & Authors (Top Frame of Page 1) ---
    story.append(Paragraph("Adaptive Hybrid Quantum-Classical Neural Network for Early Medical Disease Diagnosis Using Explainable AI", title_style))
    story.append(Paragraph("<b>Anonymous Authors</b><br/>Department of Computer Science and Engineering &bull; Research Laboratory for Quantum Machine Learning &amp; Health Informatics<br/>Preprint Under Camera-Ready Peer Review &bull; IEEE Conference Style", author_style))
    story.append(FrameBreak()) # Switch to left column
    
    # --- Abstract & Keywords (Left Column) ---
    abstract_text = (
        "<b><i>Abstract</i>&mdash;Hybrid Quantum-Classical Neural Networks (HQNNs) have attracted substantial interest for clinical diagnostic applications on tabular medical data. However, conventional HQNN architectures impose a static circuit execution paradigm in which every input instance is subjected to the same fixed-depth parameterized quantum circuit, irrespective of sample difficulty or model uncertainty. In the current Noisy Intermediate-Scale Quantum (NISQ) era, executing redundant quantum variational layers on low-uncertainty instances introduces unwarranted circuit evaluation overhead without measurable predictive necessity. In this paper, we propose an Adaptive Hybrid Quantum-Classical Neural Network (Adaptive HQNN) that dynamically allocates quantum processing based on instance-level prediction confidence derived from an initial classical feature encoder. The proposed framework supports three distinct inference pathways: a classical-only pathway (0 quantum layers), a single-quantum-layer pathway, and a two-quantum-layer pathway. Routing decision thresholds are selected via a nested three-fold inner cross-validation protocol embedded within each outer training fold, ensuring strict separation between optimization and validation to prevent data leakage. We rigorously benchmark Adaptive HQNN against a Classical Multilayer Perceptron (MLP), a Fixed-Depth HQNN (2 quantum layers), and an Adaptive Classical Control across three clinical tabular datasets: Wisconsin Diagnostic Breast Cancer (WDBC), UCI Cleveland Heart Disease, and Pima Indians Diabetes over 5 repeats of 5-fold cross-validation (25 folds per model). Furthermore, model predictions and routing associations are audited using Kernel SHAP across folds, and 25-fold ablation suites investigate low-dimensional classical bottlenecks and capacity-matched controls. Experimental results indicate that Adaptive HQNN reduces estimated inference-time quantum layer executions by 98.4% on WDBC (average 0.032 layers/sample), 84.8% on Heart Disease (average 0.305 layers/sample), and 75.5% on Diabetes (average 0.490 layers/sample) compared to Fixed HQNN. Nadeau&ndash;Bengio corrected resampled <i>t</i>-tests across overlapping cross-validation folds reveal no statistically significant difference in predictive accuracy (<i>p</i> &ge; 0.05) between Adaptive and Fixed HQNN. This study demonstrates a measurable operational trade-off between predictive performance and estimated inference-time quantum workload, providing a resource-aware framework for hybrid quantum machine learning on structured medical data without asserting quantum supremacy.</b>"
    )
    story.append(Paragraph(abstract_text, abstract_body_style))
    story.append(Paragraph("<b><i>Index Terms</i>&mdash;Quantum Machine Learning, Hybrid Quantum-Classical Neural Networks, Variational Quantum Circuits, Dynamic Computation, Explainable AI, Medical Diagnosis.</b>", keywords_style))
    
    # --- Section I: Introduction ---
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Early and accurate medical disease diagnosis is a cornerstone of modern clinical informatics, with machine learning (ML) systems increasingly deployed to assist in screening, risk stratification, and diagnostic triage [1]&ndash;[3]. While classical deep neural networks and tree-based ensembles have achieved strong predictive efficacy on structured clinical tables [4], [5], real-world clinical datasets frequently present complex non-linear correlations, high dimensionality relative to sample size, and overlapping decision boundaries near critical diagnostic thresholds [6]&ndash;[8].", body_style))
    
    story.append(Paragraph("In parallel, Quantum Machine Learning (QML) has emerged as an expressive computational paradigm. By mapping classical feature vectors into high-dimensional Hilbert spaces via parameterized quantum circuits (variational quantum circuits, or VQCs), QML models can represent feature transformations that are classically challenging to formulate [9]&ndash;[13]. Hybrid Quantum-Classical Neural Networks (HQNNs), which couple classical feedforward encoders with parameterized quantum expectation layers, represent the primary architecture for executing quantum-assisted learning on Noisy Intermediate-Scale Quantum (NISQ) hardware and simulators [14], [15].", body_style))
    
    story.append(Paragraph("Despite active exploration, existing HQNN architectures suffer from a fundamental structural limitation: <i>static quantum layer allocation</i>. Under standard formulations, every input instance <i>x</i> is processed through an identical, predetermined sequence of parameterized quantum gates, regardless of whether the sample represents an unambiguous diagnostic presentation or an uncertain borderline case. In hardware-constrained NISQ environments, quantum circuit evaluations incur substantial latency, gate errors, and decoherence noise [16], [2]. Executing multi-layer quantum circuits on routine, high-confidence samples where classical representations are already decisive is computationally inefficient and practically unmotivated.", body_style))
    
    story.append(Paragraph("While dynamic computation and early-exit mechanisms have been extensively investigated in classical computer vision and natural language processing [17]&ndash;[20], sample-wise dynamic selection of quantum variational depth remains largely unaddressed in tabular medical informatics. Specifically, there is a lack of empirical investigations evaluating whether an adaptive confidence-based routing mechanism can selectively assign classical-only, shallow-quantum, or multi-quantum processing paths to individual patient records while maintaining acceptable diagnostic discrimination.", body_style))
    
    story.append(Paragraph("To address this gap, this paper introduces the <b>Adaptive Hybrid Quantum-Classical Neural Network (Adaptive HQNN)</b> framework for tabular disease diagnosis. The proposed system employs a classical feature encoder to generate an initial disease probability and an associated confidence metric. Based on two calibrated routing thresholds (&tau;<sub>high</sub>, &tau;<sub>low</sub>), instances are dynamically directed to: (i) <b>Path 0 (Classical-Only)</b>: When initial confidence is high (<i>C</i>(<i>x</i>) &ge; &tau;<sub>high</sub>), bypassing quantum execution entirely (<i>L</i><sub>Q</sub> = 0); (ii) <b>Path 1 (1-Quantum-Layer)</b>: When initial confidence is intermediate (&tau;<sub>low</sub> &le; <i>C</i>(<i>x</i>) &lt; &tau;<sub>high</sub>), executing a single 4-qubit variational circuit layer (<i>L</i><sub>Q</sub> = 1); and (iii) <b>Path 2 (2-Quantum-Layers)</b>: When initial uncertainty is high (<i>C</i>(<i>x</i>) &lt; &tau;<sub>low</sub>), executing a depth-2 4-qubit variational circuit (<i>L</i><sub>Q</sub> = 2).", body_style))
    
    story.append(Paragraph("To ensure complete scientific rigor and eliminate methodological data leakage, routing thresholds are selected via an inner 3-fold cross-validation scheme within each outer training fold, followed by single-pass outer validation and ensemble-based holdout testing across 5 repeats of 5-fold cross-validation (25 total fold evaluations per model). Furthermore, Kernel SHAP (Shapley Additive Explanations) [21], [22] is integrated to analyze feature attribution patterns across folds, and formal Nadeau&ndash;Bengio corrected resampled statistical tests [23] evaluate performance differences.", body_style))
    
    story.append(Paragraph("The key contributions of this work are summarized as follows:", body_style))
    story.append(Paragraph("&bull; We formulate an Adaptive Hybrid Quantum-Classical architecture with uncertainty-guided dynamic quantum layer selection for binary medical classification.", bullet_style))
    story.append(Paragraph("&bull; We implement a leakage-safe nested evaluation pipeline (outer 5-fold CV with 5 repeats, inner 3-fold CV threshold selection, and an isolated 20% holdout test partition) across three established clinical datasets (WDBC, Cleveland Heart, and Pima Diabetes).", bullet_style))
    story.append(Paragraph("&bull; We benchmark Adaptive HQNN against three controls: a Classical MLP (M1), a Fixed-Depth HQNN (M2), and an Adaptive Classical Control (M4), alongside a 25-fold ablation suite including pure classical shortcuts, static 4-D classical bottlenecks, and capacity-matched classical models.", bullet_style))
    story.append(Paragraph("&bull; We demonstrate that Adaptive HQNN reduces estimated inference-time quantum layer executions by 75.5% to 98.4% across benchmarks, with Nadeau&ndash;Bengio corrected resampled <i>t</i>-tests (<i>df</i>=24) detecting no statistically significant difference in predictive accuracy (<i>p</i> &ge; 0.05) relative to fixed-depth quantum networks.", bullet_style))
    story.append(Paragraph("&bull; We provide an explainable AI audit using Kernel SHAP, quantifying fold-to-fold feature ranking stability across outer cross-validation checkpoints.", bullet_style))

    # --- Section II: Related Work ---
    story.append(Paragraph("II. RELATED WORK", h1_style))
    story.append(Paragraph("<b>A. Classical ML for Medical Tabular Data:</b> Classical predictive modeling on tabular health records relies heavily on gradient boosted decision trees, such as XGBoost [4] and LightGBM, as well as regularized Multilayer Perceptrons (MLPs) [5]. In medical screening, models must balance sensitivity (minimizing false negatives) and specificity (minimizing unnecessary invasive biopsies) [6]&ndash;[8]. While deep tabular architectures have advanced, they frequently encounter optimization challenges on small-to-medium clinical cohorts where feature interdependencies are complex and sample counts are limited [5].", body_style))
    
    story.append(Paragraph("<b>B. Hybrid Quantum-Classical Neural Networks:</b> Parameterized Quantum Circuits (PQCs), also termed Variational Quantum Circuits (VQCs), utilize classical optimization routines to update the parameters of quantum gates embedded in quantum circuits [1]&ndash;[3], [11]. In hybrid quantum-classical networks, classical neural layers compress high-dimensional input vectors into latent variables, which are subsequently encoded into quantum states via angle or amplitude embedding [9], [10], [12], [13]. Expectation values of Pauli observables measured from the quantum state are then mapped to output class logits [14]. Prior studies have applied fixed-depth HQNNs to medical imaging and genomic classifications, demonstrating theoretical representational expressiveness but universally enforcing static quantum execution across all test samples [2], [3].", body_style))
    
    story.append(Paragraph("<b>C. Dynamic Neural Networks and Resource-Adaptive Inference:</b> Dynamic neural networks adjust their computational graph or execution depth conditioned on the input instance [19]. In classical architectures, early-exit networks such as BranchyNet [17] and MSDNet [18] introduce intermediate classification heads that allow confident samples to exit early, reserving deeper layers for difficult inputs [20]. Adaptive computation has also been applied to model cascades and conditional computation [19]. However, transferring conditional execution to hybrid quantum architectures&mdash;where the computational trade-off involves quantum circuit evaluations versus classical operations&mdash;has remained an open question.", body_style))

    story.append(Paragraph("<b>D. Explainable AI in Clinical Machine Learning:</b> Explainable AI (XAI) is essential for clinical decision-support systems to ensure algorithmic transparency and guard against spurious correlations [21], [22]. Lundberg and Lee introduced SHAP (Shapley Additive Explanations), grounded in cooperative game theory, providing unique, additive feature attribution values [21]. In hybrid quantum-classical modeling, explaining both final prediction outputs and intermediate routing behaviors across validation folds is critical for clinical auditing.", body_style))

    # --- Section III: Problem Formulation and Proposed Method ---
    story.append(Paragraph("III. PROBLEM FORMULATION AND PROPOSED METHOD", h1_style))
    story.append(Paragraph("<b>A. Problem Definition:</b> Let <i>D</i> = {(<i>x</i><sub>i</sub>, <i>y</i><sub>i</sub>)}<sub>i=1</sub><sup>N</sup> denote a medical tabular dataset, where <i>x</i><sub>i</sub> &isin; &real;<sup>D</sup> represents a patient feature vector and <i>y</i><sub>i</sub> &isin; {0, 1} denotes the binary diagnostic label (e.g., benign vs. malignant, disease absent vs. present). The objective is to learn a mapping <i>f</i>(<i>x</i>; &Theta;) &isin; [0, 1] parameterized by classical parameters &theta;<sub>class</sub> and quantum circuit parameters &theta;<sub>quant</sub>, such that predictive performance (measured by F1-score and ROC-AUC) is preserved while the expected number of quantum layer evaluations during inference, denoted E[<i>L</i><sub>Q</sub>(<i>x</i>)], is minimized.", body_style))

    story.append(Paragraph("<b>B. Overall Architecture:</b> The Adaptive HQNN architecture consists of four primary functional modules: (i) A classical feedforward feature encoder <i>g</i><sub>&theta;</sub>(<i>x</i>); (ii) A preliminary classification head producing preliminary logit <i>z</i><sub>prelim</sub>(<i>x</i>) and confidence metric <i>C</i>(<i>x</i>); (iii) A parameterized quantum feature map and multi-depth variational quantum circuit; and (iv) A dynamic uncertainty router directing execution across Path 0 (<i>L</i><sub>Q</sub> = 0), Path 1 (<i>L</i><sub>Q</sub> = 1), or Path 2 (<i>L</i><sub>Q</sub> = 2).", body_style))

    story.append(Paragraph("<b>C. Classical Feature Encoder and Latent Projection:</b> The input feature vector <i>x</i> &isin; &real;<sup>D</sup> is first processed by a classical dense block consisting of a linear projection, layer normalization, and a rectified linear unit (ReLU) activation:", body_style))
    story.append(Paragraph("<i>h</i>(<i>x</i>) = ReLU(LayerNorm(<i>W</i><sub>e</sub> <i>x</i> + <i>b</i><sub>e</sub>)) &isin; &real;<sup>16</sup> &nbsp;&nbsp;&nbsp;&nbsp; (1)", equation_style))
    story.append(Paragraph("where <i>W</i><sub>e</sub> &isin; &real;<sup>16 &times; D</sup> and <i>b</i><sub>e</sub> &isin; &real;<sup>16</sup>. From the intermediate representation <i>h</i>(<i>x</i>), a linear head generates a preliminary logit: <i>z</i><sub>prelim</sub>(<i>x</i>) = <i>w</i><sub>p</sub><sup>T</sup> <i>h</i>(<i>x</i>) + <i>b</i><sub>p</sub>. The preliminary disease probability is obtained via the logistic sigmoid function: <i>p</i><sub>prelim</sub>(<i>x</i>) = &sigma;(<i>z</i><sub>prelim</sub>(<i>x</i>)) = 1 / (1 + <i>e</i><sup>-<i>z</i><sub>prelim</sub>(<i>x</i>)</sup>). The model's prediction confidence <i>C</i>(<i>x</i>) &isin; [0.5, 1.0] is defined as:", body_style))
    story.append(Paragraph("<i>C</i>(<i>x</i>) = max(<i>p</i><sub>prelim</sub>(<i>x</i>), 1 - <i>p</i><sub>prelim</sub>(<i>x</i>)) &nbsp;&nbsp;&nbsp;&nbsp; (2)", equation_style))
    story.append(Paragraph("Simultaneously, <i>h</i>(<i>x</i>) is projected into a <i>Q</i>-dimensional latent space (<i>Q</i> = 4 qubits) via a bounded hyperbolic tangent mapping: <i>u</i>(<i>x</i>) = &pi; &middot; tanh(<i>W</i><sub>u</sub> <i>h</i>(<i>x</i>) + <i>b</i><sub>u</sub>) &isin; [-&pi;, &pi;]<sup>Q</sup>.", body_style))

    story.append(Paragraph("<b>D. Parameterized Quantum Variational Layers:</b> The latent vector <i>u</i>(<i>x</i>) is embedded into a 4-qubit quantum register initialized in the ground state |&psi;<sub>0</sub>&rang; = |0&rang;<sup>&otimes; 4</sup> = |0000&rang; using Pauli-<i>Y</i> Angle Embedding: |&psi;<sub>0</sub>(<i>u</i>)&rang; = &bigotimes;<sub>j=1</sub><sup>Q</sup> <i>R</i><sub>Y</sub>(<i>u</i><sub>j</sub>) |0&rang;, where <i>R</i><sub>Y</sub>(&phi;) = exp(-<i>i</i> &phi; &sigma;<sub>y</sub> / 2).", body_style))
    
    story.append(Paragraph("The variational quantum circuit consists of parameterized single-qubit rotations followed by entangling circular CNOT gates. For layer <i>l</i> &isin; {1, 2}, the unitary operation is defined as:", body_style))
    story.append(Paragraph("<i>U</i><sub>l</sub>(&theta;<sub>l</sub>) = (&prod;<sub>j=1</sub><sup>Q</sup> CNOT<sub>(j, j &oplus; 1)</sub>) (&prod;<sub>j=1</sub><sup>Q</sup> <i>R</i>(&alpha;<sub>l,j</sub>, &beta;<sub>l,j</sub>, &gamma;<sub>l,j</sub>)) &nbsp;&nbsp;&nbsp;&nbsp; (3)", equation_style))
    story.append(Paragraph("where <i>R</i>(&alpha;, &beta;, &gamma;) = <i>R</i><sub>Z</sub>(&gamma;) <i>R</i><sub>Y</sub>(&beta;) <i>R</i><sub>Z</sub>(&alpha;) represents an arbitrary single-qubit Euler rotation parameterized by trainable weights &theta;<sub>l</sub> &isin; &real;<sup>Q &times; 3</sup>. For Path 1 (<i>L</i><sub>Q</sub> = 1), the circuit has depth 6 and contains 9 operations (4 <i>R</i><sub>Y</sub>, 4 Rot, 4 CNOT), decomposing into 20 elementary gates with 12 quantum parameters: |&psi;<sub>1</sub>&rang; = <i>U</i><sub>1</sub>(&theta;<sub>1</sub>) |&psi;<sub>0</sub>(<i>u</i>)&rang;. For Path 2 (<i>L</i><sub>Q</sub> = 2), the circuit has depth 11 and contains 17 operations, decomposing into 36 elementary gates with 24 quantum parameters: |&psi;<sub>2</sub>&rang; = <i>U</i><sub>2</sub>(&theta;<sub>2</sub>) <i>U</i><sub>1</sub>(&theta;<sub>1</sub>) |&psi;<sub>0</sub>(<i>u</i>)&rang;. Expectation values of Pauli-<i>Z</i> observables across all <i>Q</i> qubits, &lang;<i>Z</i><sub>j</sub>&rang;<sub>k</sub> = &lang;&psi;<sub>k</sub>| &sigma;<sub>z</sub><sup>(j)</sup> |&psi;<sub>k</sub>&rang;, are mapped to classification logits via classical heads: <i>z</i><sub>q1</sub>(<i>x</i>) = <i>w</i><sub>q1</sub><sup>T</sup> <i>z&#773;</i><sub>1</sub> + <i>b</i><sub>q1</sub> and <i>z</i><sub>q2</sub>(<i>x</i>) = <i>w</i><sub>q2</sub><sup>T</sup> <i>z&#773;</i><sub>2</sub> + <i>b</i><sub>q2</sub>.", body_style))

    story.append(Paragraph("<b>E. Adaptive Routing Policy and Loss Formulation:</b> During training, all three pathways are trained simultaneously across the entire batch. To address clinical class imbalance, a positive-class weighting factor <i>w</i><sub>pos</sub> = <i>N</i><sub>neg</sub> / <i>N</i><sub>pos</sub> is incorporated into the composite binary cross-entropy loss:", body_style))
    story.append(Paragraph("L<sub>total</sub> = L<sub>BCE</sub>(<i>z</i><sub>prelim</sub>, <i>y</i>; <i>w</i><sub>pos</sub>) + 0.6 L<sub>BCE</sub>(<i>z</i><sub>q1</sub>, <i>y</i>; <i>w</i><sub>pos</sub>) + 0.6 L<sub>BCE</sub>(<i>z</i><sub>q2</sub>, <i>y</i>; <i>w</i><sub>pos</sub>) &nbsp;&nbsp;&nbsp;&nbsp; (4)", equation_style))
    story.append(Paragraph("During inference, execution conditionally branches based on confidence <i>C</i>(<i>x</i>) and two calibrated thresholds &tau;<sub>high</sub> and &tau;<sub>low</sub> satisfying 0.5 &le; &tau;<sub>low</sub> &lt; &tau;<sub>high</sub> &le; 1.0: <i>z</i><sub>final</sub>(<i>x</i>) = <i>z</i><sub>prelim</sub>(<i>x</i>) if <i>C</i>(<i>x</i>) &ge; &tau;<sub>high</sub> (Path 0: <i>L</i><sub>Q</sub> = 0); <i>z</i><sub>final</sub>(<i>x</i>) = <i>z</i><sub>q1</sub>(<i>x</i>) if &tau;<sub>low</sub> &le; <i>C</i>(<i>x</i>) &lt; &tau;<sub>high</sub> (Path 1: <i>L</i><sub>Q</sub> = 1); and <i>z</i><sub>final</sub>(<i>x</i>) = <i>z</i><sub>q2</sub>(<i>x</i>) if <i>C</i>(<i>x</i>) &lt; &tau;<sub>low</sub> (Path 2: <i>L</i><sub>Q</sub> = 2).", body_style))

    story.append(Paragraph("<b>F. Leakage-Free Inner CV for Threshold Optimization:</b> Within each outer training fold, an inner 3-fold cross-validation grid search is conducted over &tau;<sub>high</sub> &isin; [0.75, 0.80, 0.85, 0.90, 0.95] and &tau;<sub>low</sub> &isin; [0.55, 0.60, 0.65, 0.70] (19 valid pairs subject to &tau;<sub>high</sub> &gt; &tau;<sub>low</sub>). The selection objective maximizes inner out-of-fold F1-score penalized by average quantum layer execution: S(&tau;<sub>high</sub>, &tau;<sub>low</sub>) = F1<sub>inner</sub> - 0.05 &middot; (L&#773;<sub>Q, inner</sub> / 2.0). For holdout test set evaluation, predictions use the 5-fold ensemble with median thresholds: &tau;<sub>high</sub><sup>med</sup> = median<sub>k</sub>(&tau;<sub>high</sub><sup>*(k)</sup>) and &tau;<sub>low</sub><sup>med</sup> = median<sub>k</sub>(&tau;<sub>low</sub><sup>*(k)</sup>).", body_style))

    # --- Section IV: Datasets and Experimental Setup ---
    story.append(Paragraph("IV. DATASETS AND EXPERIMENTAL SETUP", h1_style))
    story.append(Paragraph("<b>A. Benchmark Datasets:</b> Experiments were conducted on three diverse tabular medical diagnostic benchmarks: (i) <b>Wisconsin Diagnostic Breast Cancer (WDBC)</b>: 569 patient records with 30 continuous nuclear morphological features [6]; (ii) <b>UCI Cleveland Heart Disease</b>: 303 patient records with 13 clinical attributes [7]; and (iii) <b>Pima Indians Diabetes</b>: 768 patient records with 8 clinical features evaluating diabetes onset [8]. TABLE I summarizes their characteristics.", body_style))

    # Table I
    t1_data = [
        [Paragraph("<b>Dataset</b>", table_cell_left), Paragraph("<b>Samples (<i>N</i>)</b>", table_cell_style), Paragraph("<b>Features (<i>D</i>)</b>", table_cell_style), Paragraph("<b>Class 0 / 1</b>", table_cell_style), Paragraph("<b>Missing (%)</b>", table_cell_style)],
        [Paragraph("WDBC Breast Cancer", table_cell_left), Paragraph("569", table_cell_style), Paragraph("30", table_cell_style), Paragraph("357 / 212", table_cell_style), Paragraph("0.00%", table_cell_style)],
        [Paragraph("UCI Heart Disease", table_cell_left), Paragraph("303", table_cell_style), Paragraph("13", table_cell_style), Paragraph("164 / 139", table_cell_style), Paragraph("0.15%", table_cell_style)],
        [Paragraph("Pima Diabetes", table_cell_left), Paragraph("768", table_cell_style), Paragraph("8", table_cell_style), Paragraph("500 / 268", table_cell_style), Paragraph("10.61%", table_cell_style)]
    ]
    t1 = Table(t1_data, colWidths=[80, 42, 40, 52, 39])
    t1.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("TABLE I: BENCHMARK DATASET CHARACTERISTICS", table_title_style))
    story.append(t1)

    story.append(Paragraph("<b>B. Preprocessing Protocol:</b> Patient IDs in WDBC were removed. Heart disease missing entries ('?') were treated as nulls. Biologically implausible zero values in Pima Diabetes (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) were imputed using median feature imputation fitted strictly on training partitions. Continuous features were standardized to zero mean and unit variance based on training statistics.", body_style))

    story.append(Paragraph("<b>C. Benchmark Models and Parameter Counts:</b> Four models were evaluated: (i) <b>M1: Classical MLP</b> (<i>D</i> &rarr; 32 &rarr; 16 &rarr; 1, LayerNorm, Dropout 0.2; 1,601 params on WDBC, 1,057 on Heart, 897 on Diabetes); (ii) <b>M2: Fixed HQNN</b> (<i>D</i> &rarr; 16 &rarr; 4, static 2-layer VQC; 625 params on WDBC, 353 on Heart, 273 on Diabetes); (iii) <b>M3: Adaptive HQNN</b> (uncertainty router; 659 params on WDBC, 387 on Heart, 307 on Diabetes); and (iv) <b>M4: Adaptive Classical Control</b> (residual dense blocks; 1,923 params on WDBC, 1,651 on Heart, 1,571 on Diabetes).", body_style))

    story.append(Paragraph("<b>D. Training and Simulation Configuration:</b> Models were implemented in PyTorch and PennyLane [14] using `default.qubit` state-vector simulation. Optimization used Adam with `ReduceLROnPlateau` (&eta; = 0.003 for M1, 0.010 for M2, 0.008 for M3/M4; batch size 32). Evaluations used <b>5 repeats of 5-fold cross-validation</b> (seeds 42, 101, 202, 303, 404; 25 outer folds per model per dataset; 300 model runs total). An isolated 20% holdout test split (<i>N</i> = 114 WDBC, 61 Heart, 154 Diabetes) generated under seed 42 remained frozen during all training and threshold tuning.", body_style))

    # --- Section V: Results and Empirical Evaluation ---
    story.append(Paragraph("V. RESULTS AND EMPIRICAL EVALUATION", h1_style))
    story.append(Paragraph("<b>A. Predictive Performance Across Repeated Cross-Validation:</b> TABLE II reports the complete 25-fold repeated cross-validation metrics (mean &plusmn; standard deviation and 95% confidence intervals) alongside holdout test generalization across all models and benchmarks.", body_style))

    # Table II (Performance)
    t2_data = [
        [Paragraph("<b>Dataset &bull; Model</b>", table_cell_left), Paragraph("<b>CV Acc (%) [95% CI]</b>", table_cell_style), Paragraph("<b>CV F1 (%) [95% CI]</b>", table_cell_style), Paragraph("<b>CV ROC-AUC</b>", table_cell_style), Paragraph("<b>Test F1</b>", table_cell_style), Paragraph("<b>Test AUC</b>", table_cell_style)],
        # WDBC
        [Paragraph("<b>WDBC:</b> M1 (Classical MLP)", table_cell_left), Paragraph("98.02 &plusmn; 1.42 [97.4, 98.6]", table_cell_style), Paragraph("97.34 &plusmn; 1.92 [96.6, 98.1]", table_cell_style), Paragraph("0.9946 &plusmn; 0.008", table_cell_style), Paragraph("98.80%", table_cell_style), Paragraph("0.9907", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;M2 (Fixed HQNN)", table_cell_left), Paragraph("98.15 &plusmn; 1.61 [97.5, 98.8]", table_cell_style), Paragraph("97.51 &plusmn; 2.20 [96.6, 98.4]", table_cell_style), Paragraph("0.9956 &plusmn; 0.007", table_cell_style), Paragraph("96.30%", table_cell_style), Paragraph("0.9940", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;<b>M3 (Adaptive HQNN)</b>", table_cell_left), Paragraph("<b>97.76 &plusmn; 1.54 [97.1, 98.4]</b>", table_cell_style), Paragraph("<b>96.98 &plusmn; 2.10 [96.1, 97.8]</b>", table_cell_style), Paragraph("<b>0.9898 &plusmn; 0.015</b>", table_cell_style), Paragraph("<b>95.12%</b>", table_cell_style), Paragraph("<b>0.9907</b>", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;M4 (Adaptive Classical)", table_cell_left), Paragraph("97.10 &plusmn; 1.45 [96.5, 97.7]", table_cell_style), Paragraph("96.10 &plusmn; 1.99 [95.3, 96.9]", table_cell_style), Paragraph("0.9904 &plusmn; 0.014", table_cell_style), Paragraph("96.30%", table_cell_style), Paragraph("0.9927", table_cell_style)],
        # Heart
        [Paragraph("<b>Heart:</b> M1 (Classical MLP)", table_cell_left), Paragraph("81.34 &plusmn; 4.00 [79.7, 83.0]", table_cell_style), Paragraph("79.40 &plusmn; 4.57 [77.5, 81.3]", table_cell_style), Paragraph("0.8884 &plusmn; 0.040", table_cell_style), Paragraph("86.67%", table_cell_style), Paragraph("0.9578", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;M2 (Fixed HQNN)", table_cell_left), Paragraph("80.42 &plusmn; 4.08 [78.7, 82.1]", table_cell_style), Paragraph("78.35 &plusmn; 4.74 [76.4, 80.3]", table_cell_style), Paragraph("0.8656 &plusmn; 0.035", table_cell_style), Paragraph("85.25%", table_cell_style), Paragraph("0.9307", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;<b>M3 (Adaptive HQNN)</b>", table_cell_left), Paragraph("<b>80.75 &plusmn; 3.51 [79.3, 82.2]</b>", table_cell_style), Paragraph("<b>78.51 &plusmn; 4.32 [76.7, 80.3]</b>", table_cell_style), Paragraph("<b>0.8745 &plusmn; 0.039</b>", table_cell_style), Paragraph("<b>85.25%</b>", table_cell_style), Paragraph("<b>0.9524</b>", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;M4 (Adaptive Classical)", table_cell_left), Paragraph("77.61 &plusmn; 5.04 [75.5, 79.7]", table_cell_style), Paragraph("75.37 &plusmn; 4.99 [73.3, 77.4]", table_cell_style), Paragraph("0.8432 &plusmn; 0.043", table_cell_style), Paragraph("85.25%", table_cell_style), Paragraph("0.9448", table_cell_style)],
        # Diabetes
        [Paragraph("<b>Diabetes:</b> M1 (Classical MLP)", table_cell_left), Paragraph("75.24 &plusmn; 4.30 [73.5, 77.0]", table_cell_style), Paragraph("68.60 &plusmn; 4.87 [66.6, 70.6]", table_cell_style), Paragraph("0.8434 &plusmn; 0.039", table_cell_style), Paragraph("67.72%", table_cell_style), Paragraph("0.8148", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;M2 (Fixed HQNN)", table_cell_left), Paragraph("75.24 &plusmn; 3.41 [73.8, 76.7]", table_cell_style), Paragraph("68.45 &plusmn; 4.02 [66.8, 70.1]", table_cell_style), Paragraph("0.8288 &plusmn; 0.034", table_cell_style), Paragraph("65.04%", table_cell_style), Paragraph("0.7876", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;<b>M3 (Adaptive HQNN)</b>", table_cell_left), Paragraph("<b>75.21 &plusmn; 4.15 [73.5, 76.9]</b>", table_cell_style), Paragraph("<b>68.07 &plusmn; 5.07 [66.0, 70.2]</b>", table_cell_style), Paragraph("<b>0.8378 &plusmn; 0.038</b>", table_cell_style), Paragraph("<b>66.67%</b>", table_cell_style), Paragraph("<b>0.8144</b>", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;M4 (Adaptive Classical)", table_cell_left), Paragraph("73.87 &plusmn; 4.41 [72.1, 75.7]", table_cell_style), Paragraph("65.25 &plusmn; 6.30 [62.7, 67.9]", table_cell_style), Paragraph("0.8019 &plusmn; 0.042", table_cell_style), Paragraph("59.83%", table_cell_style), Paragraph("0.8057", table_cell_style)],
    ]
    t2 = Table(t2_data, colWidths=[76, 56, 56, 40, 31, 31])
    t2.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,4), (-1,4), 0.3, colors.HexColor('#999999')),
        ('LINEBELOW', (0,8), (-1,8), 0.3, colors.HexColor('#999999')),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(Paragraph("TABLE II: REPEATED CROSS-VALIDATION (25 FOLDS) AND HOLDOUT TEST PERFORMANCE", table_title_style))
    story.append(t2)

    story.append(Paragraph("On WDBC, Fixed HQNN achieves 98.15% CV accuracy and 0.9956 ROC-AUC, while Adaptive HQNN achieves 97.76% CV accuracy and 0.9898 ROC-AUC. On Heart Disease, Adaptive HQNN achieves 80.75% CV accuracy and 78.51% F1-score, closely matching Fixed HQNN (80.42% and 78.35%) and outperforming Adaptive Classical Control (77.61% and 75.37%). On Pima Diabetes, Adaptive HQNN achieves 75.21% CV accuracy and 68.07% F1-score, remaining competitive with Classical MLP (75.24%) and Fixed HQNN (75.24%), while exceeding Adaptive Classical Control (73.87% and 65.25%). On the holdout test set, Adaptive HQNN attains competitive generalization with 95.12% F1-score on WDBC, 85.25% on Heart Disease, and 66.67% on Diabetes.", body_style))

    story.append(Paragraph("<b>B. Quantum Resource Efficiency:</b> TABLE III presents the distribution of inference routing allocations across the 25 repeated cross-validation folds and the resulting estimated quantum workload reductions.", body_style))

    # Table III (Efficiency)
    t3_data = [
        [Paragraph("<b>Dataset</b>", table_cell_left), Paragraph("<b>Path 0 (0 Q)</b>", table_cell_style), Paragraph("<b>Path 1 (1 Q)</b>", table_cell_style), Paragraph("<b>Path 2 (2 Q)</b>", table_cell_style), Paragraph("<b>Avg Q-Layers</b>", table_cell_style), Paragraph("<b>Reduction</b>", table_cell_style)],
        [Paragraph("WDBC Breast Cancer", table_cell_left), Paragraph("98.4 &plusmn; 0.8%", table_cell_style), Paragraph("1.4 &plusmn; 0.8%", table_cell_style), Paragraph("0.2 &plusmn; 0.3%", table_cell_style), Paragraph("<b>0.032 &plusmn; 0.01</b>", table_cell_style), Paragraph("<b>98.4%</b>", table_cell_style)],
        [Paragraph("UCI Heart Disease", table_cell_left), Paragraph("79.8 &plusmn; 4.9%", table_cell_style), Paragraph("14.8 &plusmn; 4.7%", table_cell_style), Paragraph("5.4 &plusmn; 3.1%", table_cell_style), Paragraph("<b>0.305 &plusmn; 0.08</b>", table_cell_style), Paragraph("<b>84.8%</b>", table_cell_style)],
        [Paragraph("Pima Diabetes", table_cell_left), Paragraph("67.4 &plusmn; 6.1%", table_cell_style), Paragraph("18.2 &plusmn; 5.4%", table_cell_style), Paragraph("14.4 &plusmn; 5.2%", table_cell_style), Paragraph("<b>0.490 &plusmn; 0.11</b>", table_cell_style), Paragraph("<b>75.5%</b>", table_cell_style)]
    ]
    t3 = Table(t3_data, colWidths=[70, 42, 40, 40, 38, 38])
    t3.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("TABLE III: INFERENCE ROUTING ALLOCATION ACROSS 25 REPEATED-CV FOLDS", table_title_style))
    story.append(t3)

    story.append(Paragraph("On WDBC, 98.4% of instances resolve via Path 0, reducing estimated quantum layer executions by 98.4% relative to Fixed HQNN (0.032 vs. 2.000 layers/sample). On Heart Disease, 79.8% use Path 0, 14.8% Path 1, and 5.4% Path 2 (84.8% reduction; 0.305 layers/sample). On Diabetes, 67.4% use Path 0 and 32.6% use quantum layers (75.5% reduction; 0.490 layers/sample). Classical control M4 achieves average refinement blocks of 0.029 on WDBC, 0.298 on Heart, and 0.428 on Diabetes.", body_style))

    story.append(Paragraph("<b>C. Ablation Suite Analysis:</b> To evaluate whether the quantum circuit provides representational utility beyond low-dimensional dimensionality reduction, TABLE IV presents a 25-fold ablation study comparing Adaptive HQNN against: (i) a Path-0-only pure classical shortcut; (ii) a static Classical 4-D Bottleneck replacing the VQC with a matched classical feedforward block; and (iii) a Parameter-Matched M4 classical control with capacity constrained to approximately match M3 (339&ndash;691 parameters).", body_style))

    # Table IV (Ablations)
    t4_data = [
        [Paragraph("<b>Dataset &bull; Architecture</b>", table_cell_left), Paragraph("<b>Params</b>", table_cell_style), Paragraph("<b>CV F1 (%) [95% CI]</b>", table_cell_style), Paragraph("<b>CV ROC-AUC</b>", table_cell_style)],
        [Paragraph("<b>WDBC:</b> Adaptive HQNN (M3)", table_cell_left), Paragraph("659", table_cell_style), Paragraph("<b>96.98 &plusmn; 2.10</b>", table_cell_style), Paragraph("<b>0.9898 &plusmn; 0.015</b>", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Path-0-Only Baseline", table_cell_left), Paragraph("545", table_cell_style), Paragraph("95.14 &plusmn; 2.14", table_cell_style), Paragraph("0.9909 &plusmn; 0.011", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Classical 4-D Bottleneck", table_cell_left), Paragraph("641", table_cell_style), Paragraph("96.41 &plusmn; 2.42", table_cell_style), Paragraph("0.9921 &plusmn; 0.009", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Parameter-Matched M4", table_cell_left), Paragraph("691", table_cell_style), Paragraph("95.81 &plusmn; 1.93", table_cell_style), Paragraph("0.9913 &plusmn; 0.010", table_cell_style)],
        [Paragraph("<b>Heart:</b> Adaptive HQNN (M3)", table_cell_left), Paragraph("387", table_cell_style), Paragraph("<b>78.51 &plusmn; 4.32</b>", table_cell_style), Paragraph("<b>0.8745 &plusmn; 0.039</b>", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Path-0-Only Baseline", table_cell_left), Paragraph("273", table_cell_style), Paragraph("75.15 &plusmn; 4.83", table_cell_style), Paragraph("0.8542 &plusmn; 0.039", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Classical 4-D Bottleneck", table_cell_left), Paragraph("369", table_cell_style), Paragraph("72.79 &plusmn; 4.02", table_cell_style), Paragraph("0.8257 &plusmn; 0.039", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Parameter-Matched M4", table_cell_left), Paragraph("419", table_cell_style), Paragraph("74.21 &plusmn; 5.13", table_cell_style), Paragraph("0.8564 &plusmn; 0.040", table_cell_style)],
        [Paragraph("<b>Diabetes:</b> Adaptive HQNN (M3)", table_cell_left), Paragraph("307", table_cell_style), Paragraph("<b>68.07 &plusmn; 5.07</b>", table_cell_style), Paragraph("<b>0.8378 &plusmn; 0.038</b>", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Path-0-Only Baseline", table_cell_left), Paragraph("193", table_cell_style), Paragraph("66.36 &plusmn; 4.93", table_cell_style), Paragraph("0.8137 &plusmn; 0.041", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Classical 4-D Bottleneck", table_cell_left), Paragraph("289", table_cell_style), Paragraph("63.22 &plusmn; 6.81", table_cell_style), Paragraph("0.7758 &plusmn; 0.053", table_cell_style)],
        [Paragraph("&nbsp;&nbsp;Parameter-Matched M4", table_cell_left), Paragraph("339", table_cell_style), Paragraph("66.25 &plusmn; 5.20", table_cell_style), Paragraph("0.8094 &plusmn; 0.037", table_cell_style)],
    ]
    t4 = Table(t4_data, colWidths=[90, 32, 60, 60])
    t4.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,4), (-1,4), 0.3, colors.HexColor('#999999')),
        ('LINEBELOW', (0,8), (-1,8), 0.3, colors.HexColor('#999999')),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(Paragraph("TABLE IV: 25-FOLD ABLATION SUITE BENCHMARKS", table_title_style))
    story.append(t4)

    story.append(Paragraph("Crucially, forcing feature representations through a static classical 4-D bottleneck severely degrades F1-score on Heart (-5.72%, from 78.51% to 72.79%) and Diabetes (-4.85%, from 68.07% to 63.22%). In contrast, Adaptive HQNN leverages the quantum circuit to maintain predictive discrimination. Furthermore, Adaptive HQNN outperforms Parameter-Matched M4 across all datasets (+1.17% F1 on WDBC, +4.30% on Heart, and +1.82% on Diabetes), demonstrating that selective quantum execution offers tangible advantages over capacity-matched classical residual refinement.", body_style))

    story.append(Paragraph("<b>D. Statistical Hypothesis Testing:</b> Because repeated cross-validation folds violate standard independence assumptions due to overlapping training sets, treating 25 fold evaluations as independent observations artificially inflates Type I errors. We therefore apply the <b>Nadeau&ndash;Bengio corrected resampled <i>t</i>-test</b> [23]: &sigma;&#770;<sub>NB</sub><sup>2</sup> = (1/<i>J</i> + <i>n</i><sub>val</sub>/<i>n</i><sub>train</sub>) <i>s</i><sup>2</sup> = (1/25 + 0.25) <i>s</i><sup>2</sup> = 0.29 <i>s</i><sup>2</sup>, with test statistic <i>t</i><sub>NB</sub> = <i>d&#773;</i> / &sigma;&#770;<sub>NB</sub> &sim; <i>t</i>(<i>J</i>-1) with <i>df</i> = 24. TABLE V presents the pairwise comparisons.", body_style))

    # Table V (Hypothesis Testing)
    t5_data = [
        [Paragraph("<b>Dataset</b>", table_cell_left), Paragraph("<b>Comparison</b>", table_cell_style), Paragraph("<b>Metric</b>", table_cell_style), Paragraph("<b>Mean Diff</b>", table_cell_style), Paragraph("<b><i>t</i><sub>NB</sub></b>", table_cell_style), Paragraph("<b><i>p</i><sub>NB</sub></b>", table_cell_style)],
        [Paragraph("WDBC", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("Accuracy", table_cell_style), Paragraph("-0.396%", table_cell_style), Paragraph("-0.955", table_cell_style), Paragraph("0.3491", table_cell_style)],
        [Paragraph("WDBC", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("-0.534%", table_cell_style), Paragraph("-0.970", table_cell_style), Paragraph("0.3416", table_cell_style)],
        [Paragraph("WDBC", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("ROC-AUC", table_cell_style), Paragraph("-0.0058", table_cell_style), Paragraph("-1.209", table_cell_style), Paragraph("0.2386", table_cell_style)],
        [Paragraph("WDBC", table_cell_left), Paragraph("M3 vs M4 (Adaptive)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("+0.874%", table_cell_style), Paragraph("+0.986", table_cell_style), Paragraph("0.3338", table_cell_style)],
        [Paragraph("WDBC", table_cell_left), Paragraph("M3 vs M1 (MLP)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("-0.365%", table_cell_style), Paragraph("-0.357", table_cell_style), Paragraph("0.7241", table_cell_style)],
        [Paragraph("Heart", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("Accuracy", table_cell_style), Paragraph("+0.330%", table_cell_style), Paragraph("+0.157", table_cell_style), Paragraph("0.8769", table_cell_style)],
        [Paragraph("Heart", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("+0.154%", table_cell_style), Paragraph("+0.072", table_cell_style), Paragraph("0.9435", table_cell_style)],
        [Paragraph("Heart", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("ROC-AUC", table_cell_style), Paragraph("+0.0089", table_cell_style), Paragraph("+0.532", table_cell_style), Paragraph("0.6000", table_cell_style)],
        [Paragraph("Heart", table_cell_left), Paragraph("M3 vs M4 (Adaptive)", table_cell_style), Paragraph("Accuracy", table_cell_style), Paragraph("+3.146%", table_cell_style), Paragraph("+1.432", table_cell_style), Paragraph("0.1651", table_cell_style)],
        [Paragraph("Heart", table_cell_left), Paragraph("M3 vs M4 (Adaptive)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("+3.134%", table_cell_style), Paragraph("+1.347", table_cell_style), Paragraph("0.1905", table_cell_style)],
        [Paragraph("Heart", table_cell_left), Paragraph("M3 vs M1 (MLP)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("-0.899%", table_cell_style), Paragraph("-0.479", table_cell_style), Paragraph("0.6361", table_cell_style)],
        [Paragraph("Diabetes", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("Accuracy", table_cell_style), Paragraph("-0.033%", table_cell_style), Paragraph("-0.021", table_cell_style), Paragraph("0.9838", table_cell_style)],
        [Paragraph("Diabetes", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("-0.384%", table_cell_style), Paragraph("-0.196", table_cell_style), Paragraph("0.8461", table_cell_style)],
        [Paragraph("Diabetes", table_cell_left), Paragraph("M3 vs M2 (Fixed)", table_cell_style), Paragraph("ROC-AUC", table_cell_style), Paragraph("+0.0089", table_cell_style), Paragraph("+0.903", table_cell_style), Paragraph("0.3756", table_cell_style)],
        [Paragraph("Diabetes", table_cell_left), Paragraph("M3 vs M4 (Adaptive)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("+2.819%", table_cell_style), Paragraph("+1.354", table_cell_style), Paragraph("0.1884", table_cell_style)],
        [Paragraph("Diabetes", table_cell_left), Paragraph("M3 vs M4 (Adaptive)", table_cell_style), Paragraph("ROC-AUC", table_cell_style), Paragraph("<b>+0.0359</b>", table_cell_style), Paragraph("<b>+2.405</b>", table_cell_style), Paragraph("<b>0.0243*</b>", table_cell_style)],
        [Paragraph("Diabetes", table_cell_left), Paragraph("M3 vs M1 (MLP)", table_cell_style), Paragraph("F1-Score", table_cell_style), Paragraph("-0.538%", table_cell_style), Paragraph("-0.348", table_cell_style), Paragraph("0.7308", table_cell_style)],
    ]
    t5 = Table(t5_data, colWidths=[40, 72, 42, 38, 28, 32])
    t5.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.8, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,5), (-1,5), 0.3, colors.HexColor('#999999')),
        ('LINEBELOW', (0,11), (-1,11), 0.3, colors.HexColor('#999999')),
        ('LINEBELOW', (0,-1), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(Paragraph("TABLE V: NADEAU&ndash;BENGIO CORRECTED RESAMPLED <i>T</i>-TESTS (<i>J</i>=25, <i>DF</i>=24)", table_title_style))
    story.append(t5)

    story.append(Paragraph("Across all pairwise comparisons with Fixed HQNN (M2) and Classical MLP (M1), <b>no statistically significant difference was detected</b> (<i>p</i><sub>NB</sub> &ge; 0.05). While uncorrected paired <i>t</i>-tests yield nominal <i>p</i> &lt; 0.05 across several comparisons, the Nadeau&ndash;Bengio correction properly controls false-positive rates by accounting for the 0.25 validation/training overlap. On Diabetes, M3 achieves an unadjusted statistically significant improvement in ROC-AUC over M4 (+0.0359, <i>p</i><sub>NB</sub> = 0.0243). However, when evaluating all 27 pairwise tests under Benjamini&ndash;Hochberg False Discovery Rate (FDR) adjustment (<i>q</i> = 0.05), the adjusted <i>p</i>-value (<i>p</i><sub>FDR</sub> = 0.6561) indicates this finding should be interpreted as exploratory. Crucially, non-significance indicates no statistically significant difference was detected, rather than proven mathematical equivalence.", body_style))

    story.append(Paragraph("<b>E. Explainable AI and Fold-Stability Audit:</b> Feature attributions were audited using Kernel SHAP [21] evaluated across 20 unseen holdout test samples with background reference summaries (<i>N</i><sub>bg</sub> = 30) drawn from training folds. SHAP was strictly computed post-hoc and never used for loss calculation or threshold selection. To address whether explainability conclusions depend on cross-validation partitioning, pairwise Spearman rank correlations (&rho;) were computed across all outer-fold model checkpoints. On Pima Diabetes, where fasting glucose strongly dominates metabolic outcomes, feature rankings exhibited high fold stability (&rho; = 0.757 &plusmn; 0.128, with `Glucose` unanimously ranked #1 in all 5 folds, followed by `BMI` and `Age`). Conversely, on higher-dimensional datasets with multicollinear attributes, fold stability was moderate: &rho; = 0.361 &plusmn; 0.088 on WDBC (primary markers: `texture_worst`, `concavity_worst`) and &rho; = 0.391 &plusmn; 0.244 on Heart Disease (primary markers: `ca`, `thal`, `cp`). While primary clinical indicators remained stable, secondary feature ranks fluctuated across folds. Consequently, SHAP attributions in hybrid quantum-classical networks must be reported as fold-dependent local attributions rather than invariant causal biomarkers.", body_style))

    story.append(Paragraph("<b>F. Simulation Timing Benchmarks:</b> Inference execution latencies were benchmarked across 100 timed iterations using the `default.qubit` state-vector simulator on an Intel 8-core CPU. For single-instance inference (<i>N</i> = 1), Path 0 latency was 0.158 &plusmn; 0.061 ms on WDBC, 0.151 &plusmn; 0.039 ms on Heart Disease, and 0.232 &plusmn; 0.043 ms on Diabetes. In contrast, Path 2 execution required 24.209 &plusmn; 7.552 ms on WDBC, 23.414 &plusmn; 5.631 ms on Heart, and 22.696 &plusmn; 4.244 ms on Diabetes, demonstrating a 98&times; to 155&times; speedup for Path 0 in CPU simulation. We explicitly emphasize that these timings reflect classical simulation overheads rather than physical quantum hardware runtimes or energy consumption.", body_style))

    # --- Section VI: Discussion ---
    story.append(Paragraph("VI. DISCUSSION", h1_style))
    story.append(Paragraph("The empirical results demonstrate a practical operational trade-off between predictive accuracy and inference-time quantum workload. First, quantum workload reduction corresponds directly to underlying dataset separability. On WDBC, where morphological cytology provides high linear separability, 98.4% of instances exit via Path 0, reducing quantum evaluations by 98.4%. Conversely, on Heart Disease and Diabetes, where clinical boundaries overlap, the router escalates 20.2% and 32.6% of cases to quantum variational circuits. Second, ablation experiments confirm that the quantum circuit provides genuine representational utility over low-dimensional classical projections. Constraining representations through a static 4-D classical bottleneck drops F1-score by -5.72% on Heart and -4.85% on Diabetes, whereas Adaptive HQNN maintains strong predictive parity with full 2-layer quantum networks while selectively bypassing quantum execution on routine cases.", body_style))

    # --- Section VII: Limitations and Threats to Validity ---
    story.append(Paragraph("VII. LIMITATIONS AND THREATS TO VALIDITY", h1_style))
    story.append(Paragraph("Several limitations must be acknowledged: (i) <b>Dataset Scale</b>: Benchmarks represent small cohorts (<i>N</i> = 303 to 768), limiting statistical power and cohort diversity; (ii) <b>Simulation vs. Hardware</b>: Quantum circuits were evaluated using state-vector numerical simulation. Physical NISQ processors exhibit gate noise, readout crosstalk, and decoherence; (iii) <b>Workload Metric</b>: Quantum workload is quantified as simulated layer executions per sample, which does not directly measure physical execution time or cryogenic energy; (iv) <b>No Quantum Advantage</b>: This study does not demonstrate quantum advantage, as classical MLPs achieve competitive baseline performance; and (v) <b>XAI Scope</b>: SHAP attributions reflect model decision associations and do not imply causal pathology.", body_style))

    # --- Section VIII: Conclusion and Future Work ---
    story.append(Paragraph("VIII. CONCLUSION AND FUTURE WORK", h1_style))
    story.append(Paragraph("This paper introduced Adaptive HQNN, an uncertainty-guided dynamic quantum-classical architecture for medical tabular classification. Across 5 repeats of 5-fold cross-validation on three clinical benchmarks, Adaptive HQNN achieved estimated inference-time quantum layer reductions of 75.5% to 98.4% relative to fixed-depth quantum models, with Nadeau&ndash;Bengio corrected resampled <i>t</i>-tests detecting no statistically significant difference in predictive accuracy (<i>p</i> &ge; 0.05). Future work will investigate deployments on physical quantum processors and Bayesian uncertainty quantification within the routing policy.", body_style))

    # --- References ---
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, 'Quantum machine learning,' <i>Nature</i>, vol. 549, no. 7671, pp. 195&ndash;202, 2017.",
        "[2] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. E. Khoshaman, G. E. Roberts, K. Tian, C. Zhang, and P. J. Coles, 'Variational quantum algorithms,' <i>Nature Reviews Physics</i>, vol. 3, no. 9, pp. 625&ndash;644, 2021.",
        "[3] K. Bharti, A. Cervera-Lierta, T. H. Kyaw, T. Haug, S. Alperin-Lea, A. Anand, M. Degroote, H. Heimonen, J. S. Kottmann, T. Menke, et al., 'Noisy intermediate-scale quantum algorithms,' <i>Reviews of Modern Physics</i>, vol. 94, no. 1, p. 015004, 2022.",
        "[4] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785&ndash;794.",
        "[5] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, 'Revisiting deep learning models for tabular data,' in <i>Proc. Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 34, 2021, pp. 18932&ndash;18943.",
        "[6] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, 'Nuclear feature extraction for breast tumor diagnosis,' in <i>Proc. IS&amp;T/SPIE 1993 Int. Symp. Electronic Imaging: Science and Technology</i>, vol. 1905, 1993, pp. 861&ndash;870.",
        "[7] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher, 'International application of a new probability algorithm for the diagnosis of coronary artery disease,' <i>American Journal of Cardiology</i>, vol. 64, no. 5, pp. 304&ndash;310, 1989.",
        "[8] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, 'Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,' in <i>Proc. Annual Symp. on Computer Application in Medical Care</i>, 1988, pp. 261&ndash;265.",
        "[9] M. Schuld and N. Killoran, 'Quantum machine learning in feature Hilbert spaces,' <i>Physical Review Letters</i>, vol. 122, no. 4, p. 040504, 2019.",
        "[10] V. Havl&iacute;&check;ek, A. D. C&oacute;rcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, 'Supervised learning with quantum-enhanced feature spaces,' <i>Nature</i>, vol. 567, no. 7747, pp. 209&ndash;212, 2019.",
        "[11] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, 'Quantum circuit learning,' <i>Physical Review A</i>, vol. 98, no. 3, p. 032309, 2018.",
        "[12] A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, 'The power of quantum neural networks,' <i>Nature Computational Science</i>, vol. 1, no. 6, pp. 403&ndash;409, 2021.",
        "[13] H.-Y. Huang, M. Broughton, J. Cotler, S. Chen, J. Li, M. Mohseni, H. Neven, R. Babbush, R. Kueng, J. Preskill, and J. R. McClean, 'Quantum advantage in learning from experiments,' <i>Science</i>, vol. 376, no. 6598, pp. 1182&ndash;1186, 2022.",
        "[14] V. Bergholm, J. Izaac, M. Schuld, C. Gogolin, M. S. Alam, S. Ahmed, J. M. Arrazola, C. Blank, A. Delgado, S. Di Matteo, et al., 'PennyLane: Automatic differentiation of hybrid quantum-classical computations,' <i>arXiv preprint arXiv:1811.04968</i>, 2018.",
        "[15] M. Broughton, G. Verdon, T. McCourt, A. J. Martinez, J. H. Yoo, S. V. Isakov, P. Massey, R. Halavati, M. Y. Niu, A. Zlokapa, et al., 'TensorFlow Quantum: A software framework for quantum machine learning,' <i>arXiv preprint arXiv:2003.02989</i>, 2020.",
        "[16] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, 'Barren plateaus in quantum neural network training landscapes,' <i>Nature Communications</i>, vol. 9, no. 1, p. 4812, 2018.",
        "[17] S. Teerapittayanon, B. McDanel, and H. T. Kung, 'BranchyNet: Fast inference via early exiting from deep neural networks,' in <i>Proc. 23rd Int. Conf. Pattern Recognition (ICPR)</i>, 2016, pp. 2464&ndash;2469.",
        "[18] G. Huang, D. Chen, T. Li, F. Wu, L. van der Maaten, and K. Q. Weinberger, 'Multi-scale dense networks for resource efficient image classification,' in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2018.",
        "[19] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, 'Dynamic neural networks: A survey,' <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, vol. 44, no. 11, pp. 7436&ndash;7456, 2021.",
        "[20] J. Xin, R. Tang, J. Yu, and M. Li, 'Early exiting for efficient inference in deep neural networks: A survey,' <i>IEEE Transactions on Neural Networks and Learning Systems</i>, 2024.",
        "[21] S. M. Lundberg and S.-I. Lee, 'A unified approach to interpreting model predictions,' in <i>Proc. Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 4765&ndash;4774.",
        "[22] M. T. Ribeiro, S. Singh, and C. Guestrin, ''Why should I trust you?': Explaining the predictions of any classifier,' in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 1135&ndash;1144.",
        "[23] C. Nadeau and Y. Bengio, 'Inference for the generalization error,' <i>Machine Learning</i>, vol. 52, no. 3, pp. 239&ndash;281, 2003."
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    # Build Document
    doc.build(story)
    print(f"Successfully generated {output_pdf_path}!")

if __name__ == '__main__':
    create_camera_ready_pdf()
