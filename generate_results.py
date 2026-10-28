"""
generate_results.py  -  menghasilkan SEMUA angka, tabel, dan gambar untuk bagian
"Results and Discussion" naskah SMATIKA, memakai pipeline yang SAMA dengan pengembangan.py.

Pemakaian:
    python generate_results.py dataset_suv.csv            # K otomatis (elbow)
    python generate_results.py dataset_suv.csv --k 4      # paksa K tertentu
Output ada di folder ./hasil_jurnal/
"""
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

FEATURES = ["Hp", "Torque_Nm", "Fuel_Eco_km/l", "0_100", "Curb_W_kg"]
NAME_COLS = ["model", "name", "car", "suv_name", "vehicle"]

def load(path):
    raw = pd.read_csv(path)
    suv = raw[raw["Category"].astype(str).str.upper() == "SUV"].copy() if "Category" in raw.columns else raw.copy()
    n_suv = len(suv)
    for c in FEATURES:
        suv[c] = pd.to_numeric(suv[c], errors="coerce")
    suv = suv.dropna(subset=FEATURES).reset_index(drop=True)      # nama ikut terfilter -> tidak bergeser
    name_col = next((c for c in suv.columns if c.lower() in NAME_COLS), None)
    return raw, n_suv, suv, name_col

def elbow_knee(k_values, wcss):
    x = (np.array(k_values) - min(k_values)) / (max(k_values) - min(k_values))
    y = (np.array(wcss) - min(wcss)) / (max(wcss) - min(wcss))
    pts = np.column_stack((x, y)); line = pts[-1] - pts[0]
    d = np.abs(line[0] * (pts[0][1] - pts[:, 1]) - line[1] * (pts[0][0] - pts[:, 0])) / np.linalg.norm(line)
    return int(k_values[int(np.argmax(d))]), d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--hp", type=float, default=200); ap.add_argument("--fuel", type=float, default=12.0)
    ap.add_argument("--out", default="hasil_jurnal"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw, n_suv, df, name_col = load(a.csv)
    X = StandardScaler().fit_transform(df[FEATURES])
    max_k = min(10, len(X) - 1); ks = list(range(2, max_k + 1))
    rows = []
    for k in ks:
        m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "WCSS": m.inertia_, "Silhouette": silhouette_score(X, m.labels_),
                     "Davies_Bouldin": davies_bouldin_score(X, m.labels_)})
    val = pd.DataFrame(rows)
    k_elbow, dist = elbow_knee(ks, val["WCSS"].tolist()); val["Elbow_distance"] = dist
    k_sil = int(val.loc[val["Silhouette"].idxmax(), "K"]); k_dbi = int(val.loc[val["Davies_Bouldin"].idxmin(), "K"])
    K = a.k or k_elbow
    val.round(4).to_csv(f"{a.out}/table_validity_per_k.csv", index=False)

    model = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42).fit(X)
    df["Cluster"] = model.labels_
    prof = df.groupby("Cluster")[FEATURES].mean(); prof["Jumlah_SUV"] = df["Cluster"].value_counts().sort_index()
    prof["Persen"] = 100 * prof["Jumlah_SUV"] / len(df)
    prof.round(2).to_csv(f"{a.out}/table_cluster_profile.csv")
    df[FEATURES].describe().T.round(2).to_csv(f"{a.out}/table_descriptive.csv")
    df.to_csv(f"{a.out}/hasil_klaster_suv.csv", index=False)

    rec = df[(df["Hp"] >= a.hp) & (df["Fuel_Eco_km/l"] >= a.fuel)]
    cols = ([name_col] if name_col else []) + ["Cluster", "Hp", "Torque_Nm", "Fuel_Eco_km/l", "0_100"]
    rec[cols].to_csv(f"{a.out}/recommendation_example.csv", index=False)

    # ---- gambar ----
    plt.figure(figsize=(5.2, 3.6)); plt.plot(ks, val["WCSS"], "o-", color="#1D9BF0", linewidth=2)
    plt.axvline(k_elbow, ls="--", color="#F59E0B", label=f"Auto K = {k_elbow}")
    plt.xlabel("Number of clusters (K)"); plt.ylabel("WCSS"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(f"{a.out}/fig_elbow.png", dpi=220); plt.close()

    fig, ax1 = plt.subplots(figsize=(5.2, 3.6)); ax2 = ax1.twinx()
    ax1.plot(ks, val["Silhouette"], "o-", color="#1D9BF0", linewidth=2, label="Silhouette (higher = better)")
    ax2.plot(ks, val["Davies_Bouldin"], "s--", color="#C0504D", label="Davies–Bouldin (lower = better)")
    ax1.set_xlabel("Number of clusters (K)"); ax1.set_ylabel("Silhouette score"); ax2.set_ylabel("Davies–Bouldin index")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="best"); ax1.grid(alpha=.25); fig.tight_layout()
    plt.savefig(f"{a.out}/fig_validity.png", dpi=220); plt.close()

    colors = ["#1D9BF0", "#22C55E", "#F59E0B", "#A855F7", "#EF4444", "#38BDF8"]
    plt.figure(figsize=(5.2, 3.8))
    for c in range(K):
        s = df[df["Cluster"] == c]
        plt.scatter(s["Hp"], s["Fuel_Eco_km/l"], s=28, alpha=.8, color=colors[c % len(colors)], label=f"Cluster {c}")
    plt.xlabel("Engine power (hp)"); plt.ylabel("Fuel economy (km/l)"); plt.legend(fontsize=7); plt.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_scatter.png", dpi=220); plt.close()

    summary = {"raw_rows": len(raw), "suv_rows_before_cleaning": n_suv, "suv_rows_after_cleaning": len(df),
               "K_used": K, "K_by_elbow": k_elbow, "K_by_silhouette": k_sil, "K_by_davies_bouldin": k_dbi,
               "WCSS": round(model.inertia_, 2), "Silhouette": round(silhouette_score(X, model.labels_), 3),
               "Davies_Bouldin": round(davies_bouldin_score(X, model.labels_), 3),
               "recommendation_query": {"min_hp": a.hp, "min_fuel_km_l": a.fuel, "n_results": len(rec)}}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2)); print("\nProfil klaster:\n", prof.round(2)); print("\nValiditas per K:\n", val.round(3))

if __name__ == "__main__":
    main()
