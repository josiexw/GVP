import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

shape_map = [
    "cube_4.32_0.12.png",
    "square_2d_3.13_1.42.png",
    "cube_5.57_1.17.png",
    "hex_prism_6.2_1.52.png",
    "hex_2d_5.75_0.91.png",
    "hex_prism_0.1_0.38.png",
    "l_2d_0.72_1.44.png",
    "l_prism_6.28_0.05.png",
    "l_prism_4.0_0.14.png",
    "tri_2d_5.74_1.54.png",
    "tri_prism_4.73_0.16.png",
    "tri_2d_5.19_0.94.png",
]

WIRE_2D_IDS = {1, 4, 6, 9, 11}

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 24    ,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 28,
        "grid.alpha": 0.5,
    }
)

def load_shape_icon(shape_id, folder, size=120):
    fname = shape_map[shape_id]
    path = os.path.join(folder, fname)
    img = Image.open(path).convert("RGBA").resize((size, size))
    return np.asarray(img)

def label_barh_values(ax, bars, x_pad=0.012):
    c = bars.patches[0].get_facecolor()
    for b in bars.patches:
        w = b.get_width()
        y = b.get_y() + b.get_height() / 2
        ax.text(
            w + x_pad,
            y,
            f"{w:.2f}",
            va="center",
            ha="left",
            color=c,
            fontweight="bold",
            fontsize=10,
            clip_on=False,
        )

def parse_topk_file(filepath):
    results = []
    if not os.path.exists(filepath):
        return results
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rank = int(parts[0].split('=')[1])
            logL = float(parts[5].split('=')[1])
            filename = parts[6].split('=')[1]
            results.append({'rank': rank, 'logL': logL, 'filename': filename})
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_csv", type=str, default="results/gvp_model.csv")
    parser.add_argument("--human_csv", type=str, default="results/gvp_human.csv")
    parser.add_argument("--shape_dir", type=str, default="shapes")
    parser.add_argument("--topk_dir", type=str, default="topk_poses")
    parser.add_argument("--out_dir", type=str, default="figs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # MODEL VS HUMAN
    dfm = pd.read_csv(args.model_csv)
    dfh_raw = pd.read_csv(args.human_csv)

    for c in dfh_raw.columns:
        dfh_raw[c] = pd.to_numeric(dfh_raw[c], errors="coerce")
    human_p3d = (dfh_raw - 1.0) / 9.0

    df3 = dfm[~dfm["shape_id"].isin(WIRE_2D_IDS)].copy()
    df3 = df3.sort_values("p3d", ascending=False)
    y = np.arange(len(df3))
    h = 0.36

    fig = plt.figure(figsize=(11.2, 6.8))
    ax = plt.gca()

    bars_model = ax.barh(
        y - h / 2,
        df3["p3d"].values,
        height=h,
        label="Model",
        alpha=0.9,
    )
    bars_human = ax.barh(
        y + h / 2,
        df3["human"].values,
        height=h,
        label="Human",
        alpha=0.9,
    )

    ax.invert_yaxis()
    ax.set_xlabel("P(3D|Image)")
    ax.set_title("3D Wireframes: Model vs Human Prediction")

    ax.set_xlim(-0.14, 1.10)
    ax.set_xticks(np.linspace(0, 1, 5))

    ax.set_yticks([])
    ax.tick_params(axis="y", left=False)

    label_barh_values(ax, bars_model, x_pad=0.012)
    label_barh_values(ax, bars_human, x_pad=0.012)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.75)
    leg.get_frame().set_edgecolor("none")

    x_icon = -0.025
    for yy, sid in zip(y, df3["shape_id"].values):
        arr = load_shape_icon(int(sid), folder=args.shape_dir, size=130)
        im = OffsetImage(arr, zoom=0.46)
        ab = AnnotationBbox(im, (x_icon, yy), frameon=False, box_alignment=(1.0, 0.5))
        ax.add_artist(ab)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "3d_model_vs_human.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    df2 = dfm[dfm["shape_id"].isin(WIRE_2D_IDS)].copy()
    df2["human_p2d"] = 1.0 - df2["human"].astype(float)
    df2 = df2.sort_values("p2d", ascending=False)
    y = np.arange(len(df2))

    fig = plt.figure(figsize=(11.2, 5.2))
    ax = plt.gca()

    bars_model = ax.barh(
        y - h / 2,
        df2["p2d"].values,
        height=h,
        label="Model",
        alpha=0.9,
    )
    bars_human = ax.barh(
        y + h / 2,
        df2["human_p2d"].values,
        height=h,
        label="Human",
        alpha=0.9,
    )

    ax.invert_yaxis()
    ax.set_xlabel("P(2D|Image)")
    ax.set_title("2D Wireframes: Model vs Human Prediction")

    ax.set_xlim(-0.14, 1.10)
    ax.set_xticks(np.linspace(0, 1, 5))

    ax.set_yticks([])
    ax.tick_params(axis="y", left=False)

    label_barh_values(ax, bars_model, x_pad=0.012)
    label_barh_values(ax, bars_human, x_pad=0.012)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.75)
    leg.get_frame().set_edgecolor("none")

    x_icon = -0.025
    for yy, sid in zip(y, df2["shape_id"].values):
        arr = load_shape_icon(int(sid), folder=args.shape_dir, size=130)
        im = OffsetImage(arr, zoom=0.46)
        ab = AnnotationBbox(im, (x_icon, yy), frameon=False, box_alignment=(1.0, 0.5))
        ax.add_artist(ab)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "2d_model_vs_human.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # HUMAN DISTRIBUTIONS
    fig = plt.figure(figsize=(14.5, 12))
    for sid in range(12):
        ax = plt.subplot(4, 3, sid + 1)
        yvals = human_p3d[str(sid)].values.astype(float)
        yvals = yvals[np.isfinite(yvals)]
        ax.hist(yvals, bins=12, range=(0.0, 1.0), alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="y")
        if sid == 10:
            ax.set_xlabel("Human P(3D|Image)")

        arr = load_shape_icon(sid, folder=args.shape_dir, size=150)
        im = OffsetImage(arr, zoom=0.5)
        ab = AnnotationBbox(im, (0.5, 1.05), xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0))
        ax.add_artist(ab)

    fig.suptitle("Human P(3D|Image) Distributions", y=0.92, fontsize=28)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.1, w_pad=0.4)
    fig.savefig(os.path.join(args.out_dir, "human_p3d_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # MODEL VS HUMAN SCATTER
    mean_human = human_p3d.mean(axis=0).rename("human_mean_from_raw").reset_index()
    mean_human["shape_id"] = mean_human["index"].astype(int)
    mean_human = mean_human.drop(columns=["index"])

    dfm2 = dfm.merge(mean_human, on="shape_id", how="left")
    model_by_shape = dfm2.set_index("shape_id")["p3d"].astype(float).to_dict()

    xs = []
    ys = []
    for i in range(human_p3d.shape[0]):
        row = human_p3d.iloc[i]
        for sid in range(12):
            v = row.get(str(sid), np.nan)
            x = model_by_shape.get(sid, np.nan)
            if np.isfinite(x) and np.isfinite(v):
                xs.append(float(x))
                ys.append(float(v))

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    xm = x - x.mean()
    ym = y - y.mean()
    r = float((xm @ ym) / (np.sqrt((xm @ xm) * (ym @ ym)) + 1e-12))

    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    rho = float((rxm @ rym) / (np.sqrt((rxm @ rxm) * (rym @ rym)) + 1e-12))

    slope, intercept = np.polyfit(x, y, 1)

    fig = plt.figure(figsize=(7.4, 6.1))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, axis="both")
    ax.scatter(x, y, alpha=0.18, s=18, edgecolor="none")

    xx = np.linspace(0.0, 1.0, 200)
    yy = slope * xx + intercept
    ax.plot(xx, yy, linewidth=2.2, label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")

    ax.set_xlabel("Model P(3D|Image)")
    ax.set_ylabel("Human P(3D|Image) (per participant)")
    ax.set_title("Model-Human Alignment")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    leg = ax.legend(
        title=f"Pearson r={r:.3f}, Spearman r={rho:.3f}",
        frameon=True,
        framealpha=0.75,
        loc="lower right",
    )
    leg.get_frame().set_edgecolor("none")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "model_vs_human_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # TOP 3 POSES
    fig = plt.figure(figsize=(15, 22))
    
    for sid in range(12):
        topk_2d_file = os.path.join(args.topk_dir, str(sid), "shape_2D_top5.txt")
        topk_3d_file = os.path.join(args.topk_dir, str(sid), "shape_3D_top5.txt")
        
        topk_2d = parse_topk_file(topk_2d_file)[:3]
        topk_3d = parse_topk_file(topk_3d_file)[:3]
        
        # 2D column
        ax_2d = plt.subplot(12, 2, sid * 2 + 1)
        ax_2d.axis('off')
        ax_2d.set_xlim(0, 1)
        ax_2d.set_ylim(0, 1)
        
        arr = load_shape_icon(sid, folder=args.shape_dir, size=150)
        im = OffsetImage(arr, zoom=0.5)
        ab = AnnotationBbox(im, (0.14, 0.5), xycoords="axes fraction", frameon=True, 
                           box_alignment=(0.5, 0.5), pad=0.2)
        ab.patch.set_edgecolor('black')
        ab.patch.set_linewidth(2)
        ax_2d.add_artist(ab)
        
        # Display top3 2D poses
        for i, pose_info in enumerate(topk_2d):
            pose_path = os.path.join(args.topk_dir, str(sid), pose_info['filename'])
            if os.path.exists(pose_path):
                pose_img = Image.open(pose_path).convert("RGBA").resize((150, 150))
                pose_arr = np.asarray(pose_img)
                im = OffsetImage(pose_arr, zoom=0.5)
                x_pos = 0.42 + i * 0.22
                ab = AnnotationBbox(im, (x_pos, 0.6), xycoords="axes fraction", 
                                   frameon=True, box_alignment=(0.5, 0.5), pad=0.2)
                ab.patch.set_edgecolor('blue')
                ab.patch.set_linewidth(1.5)
                ax_2d.add_artist(ab)
                
                # Log likelihood text
                ax_2d.text(x_pos, 0.25, f"logL\n{pose_info['logL']:.2f}", 
                       transform=ax_2d.transAxes, ha='center', va='center', 
                       fontsize=18, color='blue', fontweight='bold')
        
        # 3D column
        ax_3d = plt.subplot(12, 2, sid * 2 + 2)
        ax_3d.axis('off')
        ax_3d.set_xlim(0, 1)
        ax_3d.set_ylim(0, 1)
        
        # Display top3 3D poses
        for i, pose_info in enumerate(topk_3d):
            pose_path = os.path.join(args.topk_dir, str(sid), pose_info['filename'])
            if os.path.exists(pose_path):
                pose_img = Image.open(pose_path).convert("RGBA").resize((130, 130))
                pose_arr = np.asarray(pose_img)
                im = OffsetImage(pose_arr, zoom=0.6)
                x_pos = 0.18 + i * 0.31
                ab = AnnotationBbox(im, (x_pos, 0.6), xycoords="axes fraction", 
                                   frameon=True, box_alignment=(0.5, 0.5), pad=0.1)
                ab.patch.set_edgecolor('red')
                ab.patch.set_linewidth(1.5)
                ax_3d.add_artist(ab)
                
                # Log likelihood text
                ax_3d.text(x_pos, 0.2, f"logL\n{pose_info['logL']:.2f}", 
                       transform=ax_3d.transAxes, ha='center', va='center', 
                       fontsize=18, color='red', fontweight='bold')
    
    fig.text(0.25, 0.995, "2D Poses", ha='center', va='top', fontsize=18, fontweight='bold', color='blue')
    fig.text(0.75, 0.995, "3D Poses", ha='center', va='top', fontsize=18, fontweight='bold', color='red')
    
    fig.suptitle("Model's Top-3 Similar Poses for Each Shape", fontsize=20, y=0.998)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.01, hspace=0.02, wspace=0.02)
    fig.savefig(os.path.join(args.out_dir, "top3_poses_2d_3d.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()