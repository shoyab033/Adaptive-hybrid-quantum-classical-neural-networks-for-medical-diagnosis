import pandas as pd

models = [
    ('M1 Classical MLP', 'results/metrics/m1_classical_mlp_metrics.csv'),
    ('M2 Fixed HQNN', 'results/metrics/m2_fixed_hqnn_metrics.csv'),
    ('M3 Adaptive HQNN', 'results/metrics/m3_adaptive_hqnn_metrics.csv'),
    ('M4 Adaptive Classical', 'results/metrics/m4_adaptive_classical_metrics.csv')
]

for name, path in models:
    df = pd.read_csv(path)
    test_rows = df[df['fold'] == 'TEST_ENSEMBLE']
    print(f"\n=================== {name} (TEST_ENSEMBLE) ===================")
    for _, r in test_rows.iterrows():
        print(f"{r['dataset']:15s} | Acc: {r['accuracy']*100:.2f}% | Prec: {r['precision']*100:.2f}% | Rec: {r['recall']*100:.2f}% | Spec: {r['specificity']*100:.2f}% | F1: {r['f1']*100:.2f}% | AUC: {r['roc_auc']:.4f}")
