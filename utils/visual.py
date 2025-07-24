import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score


def visual_of_r2_per_sensor(
    total_y,
    pred_features,
    globalind2localind,
    globalind2loop,
    tarposinlocal,
    figures_path: str = '/root/autodl-tmp/results/ds-tse/figures/',
):
    features_name = {
        'q': 'mainline flow',
        'v': 'mainline speed',
        'k': 'mainline density',
        'r': 'on-ramp flow',
        's': 'off-ramp flow'
    }
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    y_trues = total_y['true_trans'].numpy()
    y_preds = total_y['pred_trans'].numpy()

    for i in range(len(pred_features)):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.gca().set_aspect('equal', adjustable='box')
        fig, axs = plt.subplots(6,
                                7,
                                figsize=(11.7, 10),
                                sharex=True,
                                sharey=True)
        for j in range(42):
            if j not in globalind2localind.keys():
                continue
            jj = globalind2localind[j]
            loop = globalind2loop[j]
            ax = axs[j // 7, j % 7]
            y_true = y_trues[..., jj, i].reshape(-1)
            y_pred = y_preds[..., jj, i].reshape(-1)
            flag = (~np.isnan(y_true))
            y_pred = y_pred[flag]
            y_true = y_true[flag]
            fig_path = figures_path + f'{features_name[pred_features[i]]}.png'
            normal_scatter(y_true, y_pred, pred_features[i], ax)
            c = 'black' if jj in tarposinlocal else 'red'
            ax.set_title(f'Loop:{loop}', color=c, fontsize=6)
        # plt.tight_layout()
        fig.text(0.5, 0.04, 'True', ha='center', va='center', fontsize=10)
        fig.text(0.04,
                 0.5,
                 'Prediction',
                 ha='center',
                 va='center',
                 rotation='vertical',
                 fontsize=10)
        plt.savefig(f'{fig_path}', dpi=300)


def visual_of_r2(
    total_y,
    pred_features,
    figures_path: str = '/root/autodl-tmp/results/ds-tse/figures/',
):
    features_name = {
        'q': 'mainline flow',
        'v': 'mainline speed',
        'k': 'mainline density',
        'r': 'on-ramp flow',
        's': 'off-ramp flow'
    }
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    y_trues = total_y['true_trans'].numpy()
    y_preds = total_y['pred_trans'].numpy()

    for i in range(len(pred_features)):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.figure(figsize=(3.5, 3.5))
        y_true = y_trues[..., i].reshape(-1)
        y_pred = y_preds[..., i].reshape(-1)
        plt.scatter(y_true, y_pred, s=1, alpha=0.01, marker='o')
        if pred_features[i] == 'v':
            plt.plot([0, 100], [0, 100],
                     color='black',
                     linestyle='--',
                     linewidth=1)
        else:
            plt.plot([0, 2000], [0, 2000],
                     color='black',
                     linestyle='--',
                     linewidth=1)
        r2 = r2_score(y_true, y_pred)
        plt.text(5, 90, f'R$^2$={r2:.2f}', fontsize=6)
        plt.xlabel('True', fontsize=10)
        plt.ylabel('Prediction', fontsize=10)
        fig_path = figures_path + f'{features_name[pred_features[i]]}.png'
        plt.tight_layout()
        plt.savefig(f'{fig_path}', dpi=300)


def plot_predictions(
        data_path: str,
        pred_features: str,
        figures_path: str = '/root/autodl-tmp/results/ds-tse/figures/',
        mode='test'):
    features_name = {
        'q': 'mainline flow',
        'v': 'mainline speed',
        'k': 'mainline density',
        'r': 'on-ramp flow',
        's': 'off-ramp flow'
    }
    exp_id = data_path.split('/')[-2]
    figures_path = figures_path + exp_id + '/'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    with open(data_path, 'rb') as f:
        total_y = pickle.load(f)
        test_total_y = total_y[mode]
        y_trues = test_total_y['true_trans'].numpy()
        y_preds = test_total_y['pred_trans'].numpy()

        for i in range(len(pred_features)):
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.gca().set_aspect('equal', adjustable='box')
            fig, axs = plt.subplots(6,
                                    7,
                                    figsize=(7, 7),
                                    sharex=True,
                                    sharey=True)
            for j in range(y_trues.shape[-2]):
                ax = axs[j // 7, j % 7]
                y_true = y_trues[..., j, i].reshape(-1)
                y_pred = y_preds[..., j, i].reshape(-1)
                flag = (~np.isnan(y_true))
                y_pred = y_pred[flag]
                y_true = y_true[flag]
                fig_path = figures_path + f'{features_name[pred_features[i]]}.png'
                normal_scatter(y_true, y_pred, pred_features[i], ax)
            # plt.tight_layout()
            fig.text(0.5, 0.02, 'True', ha='center', va='center', fontsize=10)
            fig.text(0.02,
                     0.5,
                     'Prediction',
                     ha='center',
                     va='center',
                     rotation='vertical',
                     fontsize=10)
            plt.savefig(f'{fig_path}', dpi=300)


def normal_scatter(y_true, y_pred, feature, ax):
    if len(y_true) == 0:
        return
    ax.scatter(y_true, y_pred, s=1, alpha=0.01, marker='o')
    if feature == 'v':
        ax.plot([0, 100], [0, 100], color='black', linestyle='--',
                linewidth=1)  # y=x 参考线
    else:
        ax.plot([0, 2000], [0, 2000],
                color='black',
                linestyle='--',
                linewidth=1)
    r2 = r2_score(y_true, y_pred)
    ax.text(5, 90, f'R$^2$={r2:.2f}', fontsize=6)  # 添加 R^2 值


def group_scatter(y_true, y_pred, figure_path):
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    r2 = r2_score(y_true, y_pred)
    # 使用 histogram2d 对数据进行聚类
    x_bins = np.linspace(min_value, max_value, 20)
    y_bins = np.linspace(min_value, max_value, 20)
    counts, xedges, yedges = np.histogram2d(y_true,
                                            y_pred,
                                            bins=[x_bins, y_bins])

    # 计算每个点的中心位置
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # 创建二维网格
    xgrid, ygrid = np.meshgrid(xcenters, ycenters)

    # 将 counts 展平，并过滤掉为零的点
    counts_flat = counts.flatten()
    nonzero_mask = counts_flat > 0
    xgrid_flat = xgrid.flatten()[nonzero_mask]
    ygrid_flat = ygrid.flatten()[nonzero_mask]
    counts_nonzero = counts_flat[nonzero_mask]
    size = 100
    alphas = np.clip(counts_nonzero / counts_nonzero.max(), 0.05, 1)
    # 设置图表风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(3.5, 3.5))  # 半栏论文图尺寸

    # 绘制散点图，大小根据密度调整
    sc = plt.scatter(
        xgrid_flat,
        ygrid_flat,
        s=size,
        #  c=counts_nonzero,
        #  cmap='viridis',
        alpha=alphas)
    # plt.colorbar(sc, label='Counts')
    plt.plot([0, 100], [0, 100], color='black', linestyle='--',
             linewidth=1)  # y=x 参考线
    plt.xlabel('True', fontsize=10)
    plt.ylabel('Pred', fontsize=10)

    plt.text(5, 90, f'R$^2$={r2:.2f}', fontsize=10)  # 添加 R^2 值
    plt.tight_layout()

    # 保存图表
    plt.savefig(f'{figure_path}', dpi=300)
    plt.close()


if __name__ == '__main__':
    data_path = '/root/autodl-tmp/results/ds-tse/prediction/STID-PEMS-12-12/best_test_prediction.pkl'
    mode = 'test'
    plot_predictions(data_path, mode=mode, pred_features='v')
