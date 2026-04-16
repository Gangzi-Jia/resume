import os
import numpy as np
import torch
from solution import SVDNet
from tqdm import tqdm


def get_device():
    """获取设备信息"""
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print(f"✅ 使用CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 未检测到CUDA设备, 使用CPU进行推理")
    return device


def read_cfg_file(file_path):
    """读取配置文件"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    info = [line.strip() for line in lines if line.strip()]
    samp_num, M, N, IQ, R = map(int, info)
    return samp_num, M, N, IQ, R


def save_to_npz(U, S, V, output_path):
    """将U, S, V矩阵保存为npz格式"""
    np.savez(output_path,
             U_out=U,
             S_out=S,
             V_out=V)


if __name__ == "__main__":
    print("🚀 === SVD模型测试脚本 (V2) ===")

    # --- 配置 ---
    DATA_PATH = "./CompetitionData2"
    PREFIX = "Round2"
    # ----------------------------------------------------------------------------
    # 核心改动 7: 指向新的模型文件
    # ----------------------------------------------------------------------------
    MODEL_PATH = "svd_model_stage2.pth"
    # 不再需要单独的统计文件路径
    # STATS_PATH = "normalization_stats_v2.npz"
    OUTPUT_DIR = "./"
    BATCH_SIZE = 128

    # --- 初始化 ---
    device = get_device()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}。请先运行 train_svd.py。")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 逐个场景进行测试 ---
    # ----------------------------------------------------------------------------
    # 核心改动 8: 简化模型加载逻辑
    # 不再需要为不同维度的场景重新加载模型，假设所有场景维度一致。
    # 如果比赛中不同场景的M, N, R不同，则需要将模型加载放入循环中。
    # 根据您提供的数据，所有场景配置相同，因此可以在循环外加载一次。
    # ----------------------------------------------------------------------------

    # 读取第一个场景的配置来初始化模型
    first_cfg_path = os.path.join(DATA_PATH, f"{PREFIX}CfgData1.txt")
    if not os.path.exists(first_cfg_path):
        raise FileNotFoundError(f"未找到配置文件: {first_cfg_path}")
    _, M, N, IQ, R = read_cfg_file(first_cfg_path)

    print("🏗️ 加载模型...")
    model = SVDNet(M=M, N=N, R=R, IQ=IQ).to(device)

    # 加载完整的 state_dict，其中包含了归一化参数
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("   - 模型及归一化参数加载成功!")
    # 打印加载的归一化参数以供验证
    print(f"   - 模型内置中位数: {model.input_median.cpu().numpy().flatten()}")
    print(f"   - 模型内置MAD: {model.input_mad.cpu().numpy().flatten()}")

    files = os.listdir(DATA_PATH)
    case_indices = sorted(list(
        set([f.split('CfgData')[-1].split('.txt')[0] for f in files if 'CfgData' in f])))

    for case_idx in case_indices:
        print(f"\n--- 正在处理案例 {case_idx} ---")

        cfg_path = os.path.join(DATA_PATH, f"{PREFIX}CfgData{case_idx}.txt")
        if not os.path.exists(cfg_path):
            continue

        # 验证当前场景配置是否与模型匹配
        _, case_M, case_N, _, case_R = read_cfg_file(cfg_path)
        if not (M == case_M and N == case_N and R == case_R):
            print(f"   - ⚠️ 警告: 案例 {case_idx} 的维度与加载的模型不匹配，跳过。")
            continue

        test_data_path = os.path.join(
            DATA_PATH, f"{PREFIX}TestData{case_idx}.npy")
        H_data_orig = np.load(test_data_path)

        print(f"   - 加载测试数据维度: {H_data_orig.shape}")

        # --- 批量推理 ---
        samp_num = H_data_orig.shape[0]
        all_U_predictions = []
        all_S_predictions = []
        all_V_predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, samp_num, BATCH_SIZE), desc=f"   - 推理案例 {case_idx}"):
                batch_indices = range(i, min(i + BATCH_SIZE, samp_num))
                # --------------------------------------------------------------------
                # 核心改动 9: 直接将原始数据送入模型
                # 模型内部会自动完成归一化
                # --------------------------------------------------------------------
                H_batch = torch.from_numpy(
                    H_data_orig[batch_indices]).float().to(device)

                U_pred, S_pred, V_pred = model(H_batch)

                # 收集预测结果
                all_U_predictions.append(U_pred.cpu().numpy())
                all_S_predictions.append(S_pred.cpu().numpy())
                all_V_predictions.append(V_pred.cpu().numpy())

        # 合并所有批次的结果
        U_final = np.concatenate(all_U_predictions, axis=0)
        S_final = np.concatenate(all_S_predictions, axis=0)
        V_final = np.concatenate(all_V_predictions, axis=0)

        # --- 保存结果为npz格式 ---
        output_filename = os.path.join(
            OUTPUT_DIR, f"{PREFIX}TestOutput{case_idx}.npz")
        save_to_npz(U_final, S_final, V_final, output_filename)
        print(f"✅ 案例 {case_idx} 的结果已保存到: {output_filename}")
        print(f"   - U维度: {U_final.shape}")
        print(f"   - S维度: {S_final.shape}")
        print(f"   - V维度: {V_final.shape}")

    print("\n🎉 --- 测试完成 ---")
    print(f"所有预测结果已保存在 '{OUTPUT_DIR}' 目录下。")
