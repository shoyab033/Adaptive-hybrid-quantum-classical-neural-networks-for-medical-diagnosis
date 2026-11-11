import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_refined_diagram(output_png_path, output_svg_path):
    fig, ax = plt.subplots(figsize=(17, 7.5), dpi=300)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Styling params
    box_ec = "black"
    box_fc = "white"
    lw = 1.3
    arrow_props = dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=12)

    def draw_box(x, y, w, h, title, subtitle=""):
        rect = patches.Rectangle((x - w/2, y - h/2), w, h, 
                                 facecolor=box_fc, edgecolor=box_ec, linewidth=lw)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x, y + h*0.22, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="black")
            ax.text(x, y - h*0.22, subtitle, ha="center", va="center", fontsize=8.5, color="black")
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="black")
        return (x, y, w, h)

    # 1. Input Box
    draw_box(1.2, 3.75, 1.8, 1.4, "Input Data", r"$x \in \mathbb{R}^D$" + "\nWDBC | Heart | Pima")

    # Arrow 1 -> 2
    ax.annotate("", xy=(2.4, 3.75), xytext=(2.1, 3.75), arrowprops=arrow_props)

    # 2. Preprocessing
    draw_box(3.4, 3.75, 1.8, 1.4, "Leakage-Safe\nPreprocessing", "• Median Impute\n• Z-Score Scale")

    # Arrow 2 -> 3
    ax.annotate("", xy=(4.6, 3.75), xytext=(4.3, 3.75), arrowprops=arrow_props)

    # 3. Classical Feature Encoder
    draw_box(5.7, 3.75, 2.0, 1.6, "Feature Encoder", r"$h(x) = \mathrm{ReLU}(\mathrm{LN}(W_e x + b_e))$" + "\n" + r"$h(x) \in \mathbb{R}^{16}$")

    # Arrow 3 -> 4
    ax.annotate("", xy=(7.0, 3.75), xytext=(6.7, 3.75), arrowprops=arrow_props)

    # 4. Preliminary Classifier & Confidence Evaluator
    draw_box(8.1, 3.75, 2.0, 1.8, "Confidence &\nRouting Evaluator", 
             r"$z_{prelim} = w_p^T h + b_p$" + "\n" + 
             r"$p = \sigma(z_{prelim})$" + "\n" + 
             r"$C(x) = \max(p, 1-p)$" + "\n" + 
             r"Latent: $u(x) \in [-\pi, \pi]^4$")

    # Dynamic Branches Split from (9.1, 3.75)
    # Branch 0 (Top: Path 0)
    ax.plot([9.1, 9.7, 9.7], [3.75, 3.75, 5.75], color="black", lw=1.3)
    ax.annotate("", xy=(10.9, 5.75), xytext=(9.7, 5.75), arrowprops=arrow_props)
    ax.text(10.3, 6.1, "Path 0 ($L_Q = 0$)\n" + r"$C(x) \geq \tau_{high}^*$", fontsize=8.5, ha="center", va="bottom")

    draw_box(12.2, 5.75, 2.3, 1.3, "Classical Direct Exit", 
             r"$z_{final} = z_{prelim}(x)$" + "\n(Bypass Quantum)")

    # Branch 1 (Middle: Path 1)
    ax.annotate("", xy=(10.9, 3.75), xytext=(9.1, 3.75), arrowprops=arrow_props)
    ax.text(10.0, 4.05, "Path 1 ($L_Q = 1$)\n" + r"$\tau_{low}^* \leq C < \tau_{high}^*$", fontsize=8.5, ha="center", va="bottom")

    draw_box(12.2, 3.75, 2.3, 1.4, "1-Layer VQC ($Q=4$)", 
             r"$|\psi_1\rangle = U_1(\theta_1) |\psi_0(u)\rangle$" + "\n" + 
             r"$\vec{z}_1 = \langle Z \rangle_1 \to z_{q1}$")

    # Branch 2 (Bottom: Path 2)
    ax.plot([9.1, 9.7, 9.7], [3.75, 3.75, 1.75], color="black", lw=1.3)
    ax.annotate("", xy=(10.9, 1.75), xytext=(9.7, 1.75), arrowprops=arrow_props)
    ax.text(10.3, 1.4, "Path 2 ($L_Q = 2$)\n" + r"$C(x) < \tau_{low}^*$", fontsize=8.5, ha="center", va="top")

    draw_box(12.2, 1.75, 2.3, 1.4, "2-Layer VQC ($Q=4$)", 
             r"$|\psi_2\rangle = U_2 U_1 |\psi_0(u)\rangle$" + "\n" + 
             r"$\vec{z}_2 = \langle Z \rangle_2 \to z_{q2}$")

    # Converge from branches to Final Output (x=15.6, y=3.75)
    ax.plot([13.35, 14.1, 14.1], [5.75, 5.75, 3.75], color="black", lw=1.3)
    ax.annotate("", xy=(14.6, 3.75), xytext=(13.35, 3.75), arrowprops=arrow_props)
    ax.plot([13.35, 14.1, 14.1], [1.75, 1.75, 3.75], color="black", lw=1.3)
    ax.annotate("", xy=(14.6, 3.75), xytext=(14.1, 3.75), arrowprops=arrow_props)

    # 5. Final Output & XAI
    draw_box(15.7, 3.75, 2.0, 1.7, "Final Diagnostic\nOutput", 
             r"$p(y=1|x) = \sigma(z_{final})$" + "\n\n" + 
             "Explainable AI\n(Kernel SHAP)")

    # Title
    plt.title("Adaptive Hybrid Quantum-Classical Neural Network (Adaptive HQNN) System Architecture", 
              fontsize=12, fontweight="bold", pad=20, color="black")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08)
    plt.savefig(output_png_path, dpi=300)
    plt.savefig(output_svg_path, format="svg")
    plt.close()
    print(f"Refined diagram saved to {output_png_path}")

if __name__ == "__main__":
    import os
    out_dir = r"C:\Users\yashw\.gemini\antigravity-ide\scratch\adaptive-hqnn\results"
    os.makedirs(out_dir, exist_ok=True)
    draw_refined_diagram(
        os.path.join(out_dir, "architecture_diagram_schematic.png"),
        os.path.join(out_dir, "architecture_diagram_schematic.svg")
    )
