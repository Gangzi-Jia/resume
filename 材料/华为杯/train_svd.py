import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from solution import SVDNet
from torch import nn

# ... [get_device, read_cfg_file, calculate_ae_components, SVDLossAE 这些函数无需修改] ...


def get_device():
    if torch.cuda.is_available():
        # 这里可以指定你想用的GPU设备
        device = torch.device(
            "cuda:1" if torch.cuda.device_count() > 0 else "cuda")
        print(f"✅ 使用CUDA设备: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 未检测到CUDA设备, 使用CPU训练")
    return device


def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    info = [line.strip() for line in lines if line.strip()]
    samp_num, M, N, IQ, R = map(int, info)
    return samp_num, M, N, IQ, R


def calculate_ae_components(U_pred, S_pred, V_pred, H_true, device):
    with torch.no_grad():
        U_c = torch.complex(U_pred[..., 0], U_pred[..., 1])
        V_c = torch.complex(V_pred[..., 0], V_pred[..., 1])
        H_true_c = torch.complex(H_true[..., 0], H_true[..., 1])

        S_diag = torch.diag_embed(S_pred).to(dtype=torch.complex64)

        H_pred_c = torch.matmul(torch.matmul(
            U_c, S_diag), V_c.conj().transpose(-2, -1))

        recon_norm = torch.linalg.norm(H_true_c, 'fro', (-2, -1))
        recon_err = torch.mean(torch.linalg.norm(
            H_true_c - H_pred_c, 'fro', (-2, -1)) / (recon_norm + 1e-8))

        R_dim = U_pred.size(2)
        I = torch.eye(R_dim, dtype=torch.complex64, device=device)
        u_ortho_err = torch.mean(torch.linalg.norm(torch.matmul(
            U_c.conj().transpose(-2, -1), U_c) - I, 'fro', (-2, -1)))
        v_ortho_err = torch.mean(torch.linalg.norm(torch.matmul(
            V_c.conj().transpose(-2, -1), V_c) - I, 'fro', (-2, -1)))

        total_ae = recon_err + u_ortho_err + v_ortho_err
    return total_ae.item(), recon_err.item(), (u_ortho_err + v_ortho_err).item()


class SVDLossAE(nn.Module):
    """标准的AE损失函数，包含重建损失和正交损失。"""

    def __init__(self):
        super(SVDLossAE, self).__init__()

    def forward(self, U_pred, S_pred, V_pred, H_true, ortho_weight=1.0):
        U_c = torch.complex(U_pred[..., 0], U_pred[..., 1])
        V_c = torch.complex(V_pred[..., 0], V_pred[..., 1])
        H_true_c = torch.complex(H_true[..., 0], H_true[..., 1])
        S_diag = torch.diag_embed(S_pred).to(torch.complex64)

        H_pred_c = torch.matmul(torch.matmul(
            U_c, S_diag), V_c.conj().transpose(-2, -1))

        recon_norm = torch.linalg.norm(H_true_c, ord='fro', dim=(-2, -1))
        recon_err_num = torch.linalg.norm(
            H_true_c - H_pred_c, ord='fro', dim=(-2, -1))
        reconstruction_loss = torch.mean(recon_err_num / (recon_norm + 1e-8))

        R = U_pred.size(2)
        I = torch.eye(R, device=U_pred.device, dtype=torch.complex64)
        U_ortho_err = torch.mean(torch.linalg.norm(torch.matmul(
            U_c.conj().transpose(-2, -1), U_c) - I, 'fro', (-2, -1)))
        V_ortho_err = torch.mean(torch.linalg.norm(torch.matmul(
            V_c.conj().transpose(-2, -1), V_c) - I, 'fro', (-2, -1)))
        ortho_loss = U_ortho_err + V_ortho_err

        total_loss = reconstruction_loss + ortho_weight * ortho_loss
        return total_loss, reconstruction_loss, ortho_loss

# (修改) 将此函数内的 .view() 也替换为 .reshape()


def calculate_normalization_stats_on_gpu(H_orig_numpy, device):
    """
    在GPU上高效计算归一化所需的统计量（中位数和绝对中位差）。
    """
    print("   - (GPU加速) 计算输入归一化统计量...")
    H_orig_tensor = torch.from_numpy(H_orig_numpy).float().to(device)

    # 使用 .reshape() 替代 .view()
    flattened_H = H_orig_tensor.reshape(-1, H_orig_tensor.shape[-1])

    median, _ = torch.median(flattened_H, dim=0)
    mad, _ = torch.median(torch.abs(flattened_H - median), dim=0)

    del H_orig_tensor
    del flattened_H

    print("   - 归一化统计量计算完毕。")
    return median, mad


if __name__ == "__main__":
    print("🚀 === SVD模型训练脚本 ===")

    DATA_PATH = "/mnt/mydisk/hgjia/scr/huawei_stage2/CompetitionData2"
    PREFIX = "Round2"
    MODEL_SAVE_PATH = "svd_model_stage2.pth"

    EPOCHS = 1000
    BATCH_SIZE = 1024
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4

    VALIDATION_SPLIT = 0.1
    PATIENCE = 35
    GRAD_CLIP_NORM = 1.0

    ORTHO_WEIGHT_START = 0.06
    ORTHO_WEIGHT_END = 2.0

    device = get_device()

    print("📁 加载数据...")
    # ... [数据加载部分代码不变] ...
    all_train_data, all_train_labels, first_config = [], [], None
    files = sorted(os.listdir(DATA_PATH))
    case_indices = sorted(list(
        set([f.split('CfgData')[-1].split('.txt')[0] for f in files if 'CfgData' in f])))
    for case_idx in case_indices:
        cfg_path = os.path.join(DATA_PATH, f"{PREFIX}CfgData{case_idx}.txt")
        if not os.path.exists(cfg_path):
            continue
        _, M, N, IQ, R = read_cfg_file(cfg_path)
        if not first_config:
            first_config = {'M': M, 'N': N, 'R': R, 'IQ': IQ}
            print(f"   检测到数据尺寸: M={M}, N={N}, R={R}, IQ={IQ}")
        all_train_data.append(np.load(os.path.join(
            DATA_PATH, f"{PREFIX}TrainData{case_idx}.npy")))
        all_train_labels.append(np.load(os.path.join(
            DATA_PATH, f"{PREFIX}TrainLabel{case_idx}.npy")))
    H_orig = np.concatenate(all_train_data)
    H_true = np.concatenate(all_train_labels)
    print(f"📊 总训练样本数: {H_orig.shape[0]}")

    dataset = TensorDataset(torch.from_numpy(
        H_orig).float(), torch.from_numpy(H_true).float())
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    print("🏗️ 初始化SVD模型...")
    model_params = first_config
    model = SVDNet(**model_params).to(device)

    # if os.path.exists(MODEL_SAVE_PATH):
    #     print(f"✅ 正在加载预训练模型: '{MODEL_SAVE_PATH}'...")
    #     try:
    #         model.load_state_dict(torch.load(
    #             MODEL_SAVE_PATH, map_location=device))
    #         print("   模型加载成功，将继续训练。")
    #     except Exception as e:
    #         print(f"⚠️ 模型加载失败: {e}。将从头开始训练。")
    # else:
    #     print(f"ℹ️ 未找到预训练模型 '{MODEL_SAVE_PATH}'，将从头开始训练。")

    median, mad = calculate_normalization_stats_on_gpu(H_orig, device)

    # (修改) 使用 .reshape() 替代 .view()
    model.input_median.copy_(median.reshape(1, 1, 1, model_params['IQ']))
    model.input_mad.copy_(mad.reshape(1, 1, 1, model_params['IQ']))

    criterion = SVDLossAE().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10, verbose=True)

    print("\n🔥 === 开始训练 ===")
    # ... [训练循环部分代码不变] ...
    best_val_ae = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        progress = epoch / max(1, EPOCHS - 1)
        current_ortho_weight = ORTHO_WEIGHT_START + \
            (ORTHO_WEIGHT_END - ORTHO_WEIGHT_START) * progress

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train] O_W:{current_ortho_weight:.2f}")
        for H_in_batch, H_true_batch in progress_bar:
            H_in_batch, H_true_batch = H_in_batch.to(
                device), H_true_batch.to(device)

            optimizer.zero_grad()
            U_pred, S_pred, V_pred = model(H_in_batch)
            loss, recon_loss, ortho_loss = criterion(
                U_pred, S_pred, V_pred, H_true_batch, current_ortho_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            progress_bar.set_postfix(
                {'L': f"{loss.item():.4f}", 'Recon': f"{recon_loss.item():.4f}", 'Ortho': f"{ortho_loss.item():.4f}"})

        model.eval()
        val_ae, val_recon_ae, val_ortho_ae = 0, 0, 0
        with torch.no_grad():
            for H_in_batch, H_true_batch in val_loader:
                H_in_batch, H_true_batch = H_in_batch.to(
                    device), H_true_batch.to(device)
                U_pred, S_pred, V_pred = model(H_in_batch)

                batch_ae, batch_recon, batch_ortho = calculate_ae_components(
                    U_pred, S_pred, V_pred, H_true_batch, device)
                val_ae += batch_ae
                val_recon_ae += batch_recon
                val_ortho_ae += batch_ortho

        avg_val_ae = val_ae / len(val_loader)
        avg_recon_ae = val_recon_ae / len(val_loader)
        avg_ortho_ae = val_ortho_ae / len(val_loader)

        scheduler.step(avg_val_ae)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Val AE: {avg_val_ae:.6f} [Recon: {avg_recon_ae:.6f}, Ortho: {avg_ortho_ae:.6f}] | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_ae < best_val_ae:
            best_val_ae = avg_val_ae
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✅ Model Saved! Best Val AE: {best_val_ae:.6f}")
        else:
            patience_counter += 1
            print(
                f"  ⏳ Val AE not improved. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print(
        f"\n🎉 Training Complete. Best Val AE: {best_val_ae:.6f} saved to '{MODEL_SAVE_PATH}'")
