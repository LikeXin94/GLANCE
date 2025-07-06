import argparse
import collections
import itertools
import torch
from model_YUMC_multi_6views_20240524 import *
from get_mask_my import get_mask_my
from util import cal_std, get_logger, cal_std_my
from datasets_my import *
from configure_my_YUMC_multiple import get_default_config# 默认配置。包括缺失率，Prediction，Autoencoder，training
from sklearn.preprocessing import StandardScaler  #样本数据归一化，标准化

dataset = {
    1: "digit_6view",# 6view#[76,216,64,240,47,6][2000]
}
# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='1', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='20', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='10', help='number of test times')
parser.add_argument('--tau', type=float, default='0.1', help='hyperparameter in contrastive learning')
args = parser.parse_args()

dataset = dataset[args.dataset]

def main():
    use_cuda = torch.cuda.is_available() #检验当前GPU是否可用
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)#
    config['print_num'] = args.print_num# 100
    config['dataset'] = dataset
    config['test_time'] = args.test_time
    logger = get_logger()


    # Load data
    X_list, Y_list = load_data_my(config)
    x_train_raw1 = X_list.copy()
    view_num = len(X_list)
    Y_label = Y_list.copy()


    accumulated_metrics_lambda1 = collections.defaultdict(list)
    best_result_metrics = collections.defaultdict(list)  # 为字典提供默认值，
    paired_rate = 0# unpaired view


    args.test_time = 1
    for data_seed in range(1, args.test_time + 1):
        np.random.seed(data_seed)
        mask, tmp_idx, mis_idx = get_mask_my(view_num, Y_label, paired_rate)
        mask = torch.from_numpy(mask).long().to(device)
        a = min(len(tmp_idx[0]), len(tmp_idx[1]), len(tmp_idx[2]), len(tmp_idx[3]), len(tmp_idx[4]), len(tmp_idx[5]), )
        for v in range(view_num):
            if len(tmp_idx[v]) > a:
                tmp_idx[v] = tmp_idx[v][0:a]

        x_train_raw = []
        Y_view = []
        for v in range(0, view_num):
            a = np.array(tmp_idx[v]).astype(int)
            x_train_raw.append(x_train_raw1[v][a])
            Y_view.append(Y_label[0][a])
        Y_label_adjust = np.concatenate([Y_view[0], Y_view[1], Y_view[2], Y_view[3], Y_view[4], Y_view[5]], axis=0)

        # Set random seeds
        if config['training']['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['training']['seed']

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        COMPLETER_my = Completer_my_20220605(config)

        optimizer = torch.optim.Adam(itertools.chain(COMPLETER_my.autoencoder1.parameters(), COMPLETER_my.autoencoder2.parameters(),
                                                                COMPLETER_my.autoencoder3.parameters(), COMPLETER_my.autoencoder4.parameters(),
                                                                COMPLETER_my.autoencoder5.parameters(), COMPLETER_my.autoencoder6.parameters(),
                                                                COMPLETER_my.img2txt.parameters(), COMPLETER_my.txt2img.parameters()),
        lr=config['training']['lr'])
        COMPLETER_my.to_device(device)

        # Print the models
        logger.info(COMPLETER_my.autoencoder1)
        logger.info(COMPLETER_my.img2txt)
        logger.info(optimizer)


        standardscaler = StandardScaler()
        # 对数组x遍历，对每一个样本进行标准化
        x_train = []
        for v in range(view_num):
            scaler = standardscaler.fit(x_train_raw[v])
            x_train.append(scaler.transform(x_train_raw[v]))
        # Training
        hyper_lambda1, hyper_lambda2, hyper_lambda3 = 0, 0, 0
        acc, nmi, pre, f_measure, recall, ari, ami, acc_show, nmi_show, precision_show, F_measure_show, recall_show, ARI_show, AMI_show \
                                                = COMPLETER_my.train(config, logger, x_train, Y_label_adjust, optimizer, device,
                                                                     tmp_idx, hyper_lambda1, hyper_lambda2, hyper_lambda3,)

        accumulated_metrics_lambda1['acc'].append(acc)
        accumulated_metrics_lambda1['nmi'].append(nmi)
        accumulated_metrics_lambda1['Precision'].append(pre)
        accumulated_metrics_lambda1['F_score'].append(f_measure)
        accumulated_metrics_lambda1['Recall'].append(recall)
        accumulated_metrics_lambda1['ari'].append(ari)
        accumulated_metrics_lambda1['AMI'].append(ami)

    logger.info('--------------------Training over--------------------')
    # cal_std_my(logger, accumulated_metrics_lambda1['acc'], accumulated_metrics_lambda1['nmi'], accumulated_metrics_lambda1['Precision'],
    #                                                accumulated_metrics_lambda1['F_score'], accumulated_metrics_lambda1['Recall'],accumulated_metrics_lambda1['ari'],
    #                                                accumulated_metrics_lambda1['AMI'],)

    t_show = np.sum([acc_show, nmi_show, F_measure_show], axis=0)
    t = np.where(t_show == np.max(t_show))
    best_result_idx = list(t)[0][0]

    best_result_metrics['resul_idx'].append(best_result_idx)
    best_result_metrics['acc'].append(acc_show[best_result_idx])
    best_result_metrics['nmi'].append(nmi_show[best_result_idx])
    best_result_metrics['Precision'].append(precision_show[best_result_idx])
    best_result_metrics['F_score'].append(F_measure_show[best_result_idx])
    best_result_metrics['recall'].append(recall_show[best_result_idx])
    best_result_metrics['ARI'].append(ARI_show[best_result_idx])
    best_result_metrics['AMI'].append(AMI_show[best_result_idx])
    best_result_metrics1 = best_result_metrics
    print('best_result_metrics1', best_result_metrics1)
    exit()

if __name__ == '__main__':
    main()