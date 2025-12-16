# config.py
import torch
import os


class Config:
    """
    Configuration for SaliencyGAN
    """
    # ---------------------------
    # Device
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Path settings
    # ---------------------------
    # Dataset
    train_set_root = ".\\data\\TrainSet"
    train_stimuli_dir = os.path.join(train_set_root, "Stimuli")
    train_gt_dir = os.path.join(train_set_root, "FIXATIONMAPS")
    test_set_root = ".\\data\\TestSet"
    test_stimuli_dir = os.path.join(test_set_root, "Stimuli")
    test_gt_dir = os.path.join(test_set_root, "FIXATIONMAPS")


    eval_pred_dir = "outputs/eval_preds"
    # Save directories
    save_root = "outputs"
    model_save_dir = os.path.join(save_root, "models")
    img_save_dir = os.path.join(save_root, "images")
    log_dir = os.path.join(save_root, "logs")

    # Create dirs if not exist
    for d in [model_save_dir, img_save_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # ---------------------------
    # Training hyperparameters
    # ---------------------------
    epochs = 100
    batch_size = 8
    lr_g = 2e-4       # Generator learning rate
    lr_d = 2e-4       # Discriminator learning rate
    betas = (0.5, 0.999)

    # ---------------------------
    # Image settings
    # ---------------------------
    img_height = 160
    img_width = 320


    # Loss weights
    lambda_l1 = 50        # L1 loss weight
    lambda_adv = 1        # Adversarial loss weight

    # ---------------------------
    # Training options
    # ---------------------------
    use_l1_loss = True
    use_perceptual_loss = False   # 可自行扩展 VGG loss

    # ---------------------------
    # Evaluation settings
    # ---------------------------
    # 是否在测试阶段保存显著图
    save_prediction = True

    # Metrics for saliency (可扩展)
    metrics = ["MAE", "MSE", "CC", "NSS"]


# Export config instance
cfg = Config()
