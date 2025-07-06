import logging
import numpy as np
import math
import bisect
import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def next_batch(X1, X2, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, (i + 1))


def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
    logger.info(output)

    return

def cal_std_my(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
    elif len(arg) == 7:
        logger.info('ACC:' + str(arg[0]))
        logger.info('NMI:' + str(arg[1]))
        logger.info('Precision:' + str(arg[2]))
        logger.info('F_measure:' + str(arg[3]))
        logger.info('recall:' + str(arg[4]))
        logger.info('ARI:' + str(arg[5]))
        logger.info('AMI:' + str(arg[6]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} Precision {:.2f} std {:.2f}  F_measure {:.2f} std {:.2f}
         recall {:.2f} std {:.2f} ARI {:.2f} std {:.2f} AMI {:.2f} std {:.2f}""".format(
            np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
            np.mean(arg[1]) * 100, np.std(arg[1]) * 100,
            np.mean(arg[2]) * 100, np.std(arg[2]) * 100,
            np.mean(arg[3]) * 100, np.std(arg[3]) * 100,
            np.mean(arg[4]) * 100, np.std(arg[4]) * 100,
            np.mean(arg[5]) * 100, np.std(arg[5]) * 100,
            np.mean(arg[6]) * 100, np.std(arg[5]) * 100,
        )
    logger.info(output)
    return


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
def next_batch_multiview_flower17_3views(X1, X2, X3, batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908

        yield (batch_x1, batch_x2, batch_x3, (i + 1))# 20220908

def next_batch_multiview_digit(X1, X2, X3, X4, X5, batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, (i + 1))# 20220908

def next_batch_multiview_digit_6views(X1, X2, X3, X4, X5, X6, batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        batch_x6 = X6[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, (i + 1))# 20220908


def next_batch_multiview_flower17_7views(X1, X2, X3, X4, X5, X6, X7,batch_size):# X1, X2, X3, X4, X5, X6, config['training']['batch_size']
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]# 20220908
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]
        batch_x6 = X6[start_idx: end_idx, ...]
        batch_x7 = X7[start_idx: end_idx, ...]
        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_x7,(i + 1))# 20220908



def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""
    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0
    return warpper

def step_rampup(k):
    list_k = [3,5,10]
    i = bisect.bisect(list_k,k)
    return list_k[i-1]



def contrast_loss_clust(logits_w, logits_s, feats_ulb_w, n_clusts,
                        n_feats=256, temperature=0.5, clust_cutoff=0, device_id=0,):
    # logits_softmax = torch.softmax(logits_w.detach(), dim=-1)
    logits_softmax = logits_w
    # feats_ulb_w = feats_ulb_w.detach().cpu().numpy()

    n_clusts = int(n_clusts)
    #     print(logits_softmax.shape, feats_ulb_w.shape, feats_ulb_w.dtype, n_clusts)

    # 1) get only selected logits and feats
    max_probs, max_idx = torch.max(logits_softmax, dim=-1)
    mask_bool = max_probs.ge(clust_cutoff)  # .cpu().numpy()
    del max_probs, max_idx

    # 2) 1. kmeans
    new_pseudo, _ = kmeans(X=feats_ulb_w, num_clusters=n_clusts)
    pseudo_onehot = torch.nn.functional.one_hot(new_pseudo, num_classes=n_clusts).float()
    pseudo_onehot = pseudo_onehot.cuda()

    # 1.1 kmeans: from sklearn.cluster import KMeans

    # kmeans = KMeans(n_clusters=n_clusts)
    # # 使用K-Means模型对数据进行拟合（训练）
    # kmeans.fit(feats_ulb_w.cpu().detach().numpy())
    # # 获取每个数据点的簇标签
    # new_pseudo = kmeans.labels_
    # # print('new_pseudo', new_pseudo)
    # pseudo_onehot = torch.nn.functional.one_hot(torch.tensor(new_pseudo), num_classes=n_clusts).float()
    # pseudo_onehot = pseudo_onehot.cuda()





    # 3) get distribution for each cluster
    clust_dist = []
    for i in range(n_clusts):
        mask = new_pseudo==i
        tmp_dist_clust = logits_softmax[mask]
        if tmp_dist_clust.numel()==0:
            return torch.tensor(0.0),new_pseudo
        clust_dist.append(tmp_dist_clust.mean(0))
    dists = torch.stack(clust_dist)

    # sim = torch.mm(torch.softmax(logits_s, dim=-1), dists.t() / temperature)  # B*N, K*N --> B *K
    sim = torch.mm(logits_s, dists.t() / temperature)  # B*N, K*N --> B *K
    sim_probs = sim / sim.sum(1, keepdim=True)

    loss_c = - ((torch.log(sim_probs + 1e-6) * pseudo_onehot)).sum(1)
    # loss_c = loss_c * mask_bool.float()

    loss_c = loss_c.mean()

    return loss_c,new_pseudo


def KL_divergence_clust(logits_w, logits_s, feats_ulb_w, n_clusts,
                        n_feats=256, temperature=0.5, clust_cutoff=0, device_id=0,):
    # logits_softmax = torch.softmax(logits_w.detach(), dim=-1)
    logits_softmax = logits_w
    # feats_ulb_w = feats_ulb_w.detach().cpu().numpy()

    n_clusts = int(n_clusts)
    #     print(logits_softmax.shape, feats_ulb_w.shape, feats_ulb_w.dtype, n_clusts)

    # 1) get only selected logits and feats
    max_probs, max_idx = torch.max(logits_softmax, dim=-1)
    mask_bool = max_probs.ge(clust_cutoff)  # .cpu().numpy()
    del max_probs, max_idx

    # 2) 1. kmeans
    new_pseudo, _ = kmeans(X=feats_ulb_w, num_clusters=n_clusts)
    pseudo_onehot = torch.nn.functional.one_hot(new_pseudo, num_classes=n_clusts).float()
    pseudo_onehot = pseudo_onehot.cuda()

    # 3) get distribution for each cluster
    clust_dist = []
    clust_var = []
    for i in range(n_clusts):
        mask = new_pseudo == i
        tmp_dist_clust = logits_softmax[mask]
        if tmp_dist_clust.numel() == 0:
            return torch.tensor(0.0), new_pseudo
        clust_dist.append(tmp_dist_clust.mean(0))
        clust_var.append(tmp_dist_clust.var(0))

    dists = torch.stack(clust_dist) #不同聚类簇的距离向量。
    vars = torch.stack(clust_var) # 不同聚类簇的样本方差

    # 4) compute loss by KL-divergence
    structure_loss = 0
    for v in range(2):
        log_unreliable = F.log_softmax(logits_s, dim=1)  # for unreliable view
        p_y = F.softmax(logits_w, dim=-1)  # # reliable view
        structure_loss += F.kl_div(log_unreliable, p_y, reduction='sum')

    return structure_loss, new_pseudo, dists, vars