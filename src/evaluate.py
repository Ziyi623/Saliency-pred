import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import cfg
from utils.metric import calc_cc_score,KLD
from dataset import get_loaders
from models.generator import UNetGenerator
import os
import re
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torchvision.utils import save_image
# from utils import save_pred_vis  # write visualization helper

import matplotlib.pyplot as plt


def load_latest_generator(model, model_dir):
    """
    从目录中自动加载最新的 Generator 权重（G_epochX.pth）
    """
    files = [f for f in os.listdir(model_dir) if f.startswith("G_epoch")]
    if len(files) == 0:
        raise FileNotFoundError(f"No generator model found in {model_dir}")

    files.sort(key=lambda x: int(x.split("epoch")[1].split(".")[0]))  # 取最大 epoch
    latest = files[-1]

    path = os.path.join(model_dir, latest)
    print(f"Loading Generator model: {path}")

    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def evaluate_model():
    device = cfg.device

    # ------------------------------------------------
    # 1. 加载测试集（使用 get_loaders）
    # ------------------------------------------------
    print("Loading test dataset...")
    test_loader = get_loaders(
        cfg.test_stimuli_dir,  # ⚠️ 你需保证cfg中有 test 路径
        cfg.test_gt_dir,
        batch_size=1,
        img_size=(cfg.img_height, cfg.img_width),
    )

    # ------------------------------------------------
    # 2. 加载 Generator
    # ------------------------------------------------
    G = UNetGenerator().to(device)
    G = load_latest_generator(G, cfg.model_save_dir)
    G.eval()

    # ------------------------------------------------
    # 3. 评估：预测 + CC 得分计算
    # ------------------------------------------------
    results = []
    os.makedirs(cfg.eval_pred_dir, exist_ok=True)

    print("Evaluating...")

    with torch.no_grad():
        for idx, (imgs, gts) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(device)
            gts = gts.to(device)

            # forward
            preds = G(imgs)

            # 保存预测图，以编号命名
            save_path = os.path.join(cfg.eval_pred_dir, f"{idx:05d}.png")
            save_image(preds, save_path)

            # 原图也作对应的保存，尽管有点占地方，但方便观测实验结果
            imgs_path = os.path.join(cfg.img_save_dir,f"{idx:05d}.png")
            save_image(imgs,imgs_path)

            # numpy 转换
            pred_np = preds.squeeze().cpu().numpy()
            gt_np = gts.squeeze().cpu().numpy()

            # CC score
            cc = calc_cc_score(pred_np, gt_np)
            kld = KLD(pred_np,gt_np)
            results.append([idx,cc,kld])
    # ------------------------------------------------
    # 4. 生成表格并保存
    # ------------------------------------------------

    df = pd.DataFrame(results,columns=["filename","CC_score","KLD_score"])
    csv_path = os.path.join(cfg.save_root,"result.csv")

    df.to_csv(csv_path,index=False)
    print("\n===== Score Table =====")
    print(df)
    print(f"\nSaved table to: {csv_path}")


    plot_cc_kl_distributions(df, save_dir= cfg.save_root)


def read_losses_from_log(log_path):
    """
    读取 print_log.txt，解析 lossD, lossG, L1, Adv
    返回 dict，每一项是一个 list
    """
    lossD_list = []
    lossG_list = []
    L1_list = []
    Adv_list = []

    # 正则表达式
    pattern = re.compile(
        r"lossD:\s*([0-9.]+)\s+lossG:\s*([0-9.]+)\s+L1:\s*([0-9.]+)\s+Adv:\s*([0-9.]+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                lossD, lossG, L1, Adv = map(float, match.groups())
                lossD_list.append(lossD)
                lossG_list.append(lossG)
                L1_list.append(L1)
                Adv_list.append(Adv)

    return {
        "lossD": lossD_list,
        "lossG": lossG_list,
        "L1": L1_list,
        "Adv": Adv_list
    }



def plot_lossG(lossG):
    plt.figure()
    plt.plot(lossG)
    plt.xlabel("Iteration")
    plt.ylabel("lossG")
    plt.title("Generator Loss Curve")
    plt.grid(True)
    plt.show()

def plot_L1(L1):
    plt.figure()
    plt.plot(L1)
    plt.xlabel("Iteration")
    plt.ylabel("L1")
    plt.title("GAN L1 Curve")
    plt.grid(True)
    plt.show()

def plot_Adv(Adv):
    plt.figure()
    plt.plot(Adv)
    plt.xlabel("Iteration")
    plt.ylabel("Adv")
    plt.title("GAN Adv Curve")
    plt.grid(True)
    plt.show()

def plot_cc_kl_distributions(df, save_dir=None, show=True):
    """
    Plot:
    1) CC distribution (Histogram + KDE)
    2) KL divergence distribution (log-scale)
    3) CC vs KL joint distribution (hexbin)

    Parameters
    ----------
    df : pandas.DataFrame
        columns must include ["CC_score", "KLD_score"]
    save_dir : str or None
        if not None, save figures into this directory
    show : bool
        whether to show figures
    """

    cc = df["CC_score"].values
    kl = df["KLD_score"].values

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # Figure 1: CC distribution (Histogram + KDE)
    # ============================================================
    plt.figure(figsize=(6, 4))
    sns.histplot(cc, bins=25, kde=True, stat="density")
    plt.xlabel("CC score (higher is better)")
    plt.ylabel("Density")
    plt.title("Distribution of CC Scores")

    plt.axvline(cc.mean(), linestyle="--", label=f"Mean = {cc.mean():.3f}")
    plt.axvline(np.median(cc), linestyle=":", label=f"Median = {np.median(cc):.3f}")
    plt.legend()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "cc_distribution.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ============================================================
    # Figure 2: KL divergence distribution (log-scale)
    # ============================================================
    eps = 1e-8  # avoid log(0)
    kl_log = np.log10(kl + eps)

    plt.figure(figsize=(6, 4))
    sns.histplot(kl_log, bins=30, kde=True, stat="density")
    plt.xlabel("log10(KL divergence)")
    plt.ylabel("Density")
    plt.title("Distribution of KL Divergence (log-scale)")

    plt.axvline(kl_log.mean(), linestyle="--",
                label=f"Mean = {kl_log.mean():.3f}")
    plt.axvline(np.median(kl_log), linestyle=":",
                label=f"Median = {np.median(kl_log):.3f}")
    plt.legend()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "kl_distribution_log.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ============================================================
    # Figure 3: CC vs KL joint distribution (hexbin)
    # ============================================================
    plt.figure(figsize=(6, 5))
    hb = plt.hexbin(
        cc,
        kl,
        gridsize=40,
        bins='log',      # density in log scale
        mincnt=1
    )
    plt.colorbar(hb, label="log10(N)")
    plt.xlabel("CC score (higher is better)")
    plt.ylabel("KL divergence (lower is better)")
    plt.title("Joint Distribution of CC and KL Divergence")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "cc_vs_kl_hexbin.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

