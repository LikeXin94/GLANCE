#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'xlk'
#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'xlk'
'''
与model_my_0907一样，修改为适合多视图样本的格式；
区别： 修改于20221025，目前方法已确定，与model_my_0907相比：增加样本选择权重项（epsilon），增加增加alpha参数(视图内和视图间对比损失之间的参数)，
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
from sklearn.cluster import KMeans
from evaluation import clustering_metric
import evaluation
from util import next_batch_multiview_flower17_7views, next_batch_multiview_digit_6views# next_batch,
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import scipy.io as sio
import random
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy import linalg

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class Completer_my_20220605():
    """COMPLETER module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config
        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')# check the last dimension

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']# !!!!!!!!!!!20220907再检查,不需要

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations3'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder4 = Autoencoder(config['Autoencoder']['arch4'], config['Autoencoder']['activations4'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder5 = Autoencoder(config['Autoencoder']['arch5'], config['Autoencoder']['activations5'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder6 = Autoencoder(config['Autoencoder']['arch6'], config['Autoencoder']['activations6'],
                                        config['Autoencoder']['batchnorm'])

        # Dual predictions. 该模块不需要
        self.img2txt = Prediction(self._dims_view1)
        self.txt2img = Prediction(self._dims_view2)

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.autoencoder3.to(device)
        self.autoencoder4.to(device)
        self.autoencoder5.to(device)
        self.autoencoder6.to(device)

        self.img2txt.to(device)
        self.txt2img.to(device)



    def train(self, config, logger, x_train, Y_list, optimizer, device, tmp_idx, hyper_lambda1, hyper_lambda2,
                  hyper_lambda3):

        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x_train: data of multi-view
              Y_list: labels
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """
        view_num = len(x_train)
        for v in range(view_num):
            x_train[v] = torch.as_tensor(x_train[v]).to(device)
        train_view = x_train.copy()
        n_clusters = np.unique(Y_list)
        K = len(n_clusters)
        d_subspace = 128


        '''show loss and show measure'''
        acc_show, nmi_show, precision_show, F_measure_show = [], [], [], []
        recall_show, ARI_show, AMI_show = [], [], []


        for epoch in range(config['training']['epoch']):
            ## 随机打乱数据顺序
            shuffle_idx = np.arange(len(train_view[0]))
            X1, X2, X3, X4, X5, X6, shuffle_idx = shuffle(train_view[0], train_view[1], train_view[2], train_view[3],
                                                              train_view[4], train_view[5], shuffle_idx)
            '''中间值'''
            loss_all, loss_rec1, loss_rec2, loss_z_norm1, loss_structure1, loss_fea_structure1, loss_withY_structure1 \
                = 0, 0, 0, 0, 0, 0, 0
            loss_KL_divergence_loss1 = 0

            for batch_x1, batch_x2, batch_x3, batch_x4, batch_x5,  batch_x6, batch_No in next_batch_multiview_digit_6views(X1,
                 X2, X3, X4, X5, X6, config['training']['batch_size']):
                KL_divergence_loss = 0
                var_loss = 0


                z_before = []
                batch_x1 = batch_x1.cuda()
                z_before.append(self.autoencoder1.encoder(batch_x1))
                z_before.append(self.autoencoder2.encoder(batch_x2))
                z_before.append(self.autoencoder3.encoder(batch_x3))
                z_before.append(self.autoencoder4.encoder(batch_x4))
                z_before.append(self.autoencoder5.encoder(batch_x5))
                z_before.append(self.autoencoder6.encoder(batch_x6))

                z = []
                commonZ_term = []
                for v in range(view_num):
                    z.append(torch.as_tensor(z_before[v]).clone())
                    commonZ_term.append(torch.as_tensor(z_before[v]).clone().tolist)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_before[0]), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_before[1]), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_before[2]), batch_x3)
                recon4 = F.mse_loss(self.autoencoder4.decoder(z_before[3]), batch_x4)
                recon5 = F.mse_loss(self.autoencoder5.decoder(z_before[4]), batch_x5)
                recon6 = F.mse_loss(self.autoencoder6.decoder(z_before[5]), batch_x6)

                reconstruction_loss = recon1 + recon2 + recon3 + recon4 + recon5 + recon6
                # print('reconstruction_loss', reconstruction_loss)

                '''正交约束'''
                loss_z_norm = 0
                for v in range(view_num):
                    loss_z_norm = torch.norm(
                    z_before[v].t() @ z_before[v] - torch.as_tensor(np.eye(d_subspace).astype(np.float32)).to(
                        device), p=2)

                commonZ_term_before = []
                for v in range(view_num):
                    commonZ_term_before.append(torch.as_tensor(z_before[v]).clone())

                '''global multi-view guidance'''
                for try_time in range(1):
                    z_concat = torch.cat([z_before[0], z_before[1], z_before[2], z_before[3], z_before[4], z_before[5]], dim=0)

                    # 2.1 compute db of synthesized-view
                    estimator = KMeans(K).fit(z_concat.cpu().detach().numpy())
                    common_label_pred = estimator.labels_
                    common_DBI_avg = davies_bouldin_score(z_concat.cpu().detach().numpy(), common_label_pred)
                    tmp_commonZ_term_before = []

                    DBI,  CH_Index = [], []
                    # 2.2 compute db on each view
                    for v in range(view_num):
                        tmp_a =commonZ_term_before[v].repeat(view_num,1)
                        tmp_commonZ_term_before.append(tmp_a)
                        estimator = KMeans(K).fit(tmp_commonZ_term_before[v].cpu().detach().numpy())
                        label_pred = estimator.labels_
                        DBI1 = davies_bouldin_score(tmp_commonZ_term_before[v].cpu().detach().numpy(),label_pred)
                        DBI.append(DBI1)
                    # 2.3 compute the global multi-view guidance by KL-divergence
                    for v in range(view_num):
                        if DBI[v] <= common_DBI_avg:# DBI: the little the better
                            p_y = F.softmax(tmp_commonZ_term_before[v], dim=-1)  # for reliable view;
                            log_unreliable = F.log_softmax(z_concat, dim=1)  # for poorly view; synthesized-view
                            KL_divergence_loss += F.kl_div(log_unreliable, p_y, reduction='sum') / 1
                        if DBI[v] > common_DBI_avg:# DBI, the little the better
                            p_y = F.softmax(z_concat, dim=-1)  # # for reliable view; synthesized-view
                            log_unreliable = F.log_softmax(tmp_commonZ_term_before[v], dim=1)  # for poorly view
                            KL_divergence_loss += F.kl_div(log_unreliable, p_y, reduction='sum') / 1

                '''3 local multi-view guidance'''
                # 1) Use the Hungarian algorithm to obtain the cluster centriods matching relationship between each single view and the syntheted-view
                # 2) sort the variance on each group pairing cluster
                # 3) KL-divergence of local clusters
                '''compute common-view centroids'''
                if epoch >= 0:
                    latent_fusion_z_common = torch.cat([z_before[0], z_before[1], z_before[2], z_before[3], z_before[4], z_before[5]],dim=0)
                    estimator = KMeans(K).fit(latent_fusion_z_common.cpu().detach().numpy())
                    centroids_views_zcommon = estimator.cluster_centers_
                    label_pred_zcommon = estimator.labels_
                    # 获得不同视图之间聚类中心的匹配关系矩阵 match_centriods
                    # 3.1 Hungarian algorithm to obtain the cluster centriods matching relationship
                    h = torch.from_numpy(centroids_views_zcommon)
                    match_centriods = []
                    centroids_views, label_views = [], []
                    for v in range(view_num):
                        estimator = KMeans(K).fit(commonZ_term_before[v].cpu().detach().numpy())
                        label_pred = estimator.labels_
                        label_views.append([label_pred])
                        centroids_views.append(estimator.cluster_centers_)
                        h_tmp = torch.from_numpy(centroids_views[v])
                        Simialrity_centroids = h @ h_tmp.t() #/ config['training']['tau_cross']
                        row_ind, col_ind = linear_sum_assignment(Simialrity_centroids, maximize=True)  # 最大相似度
                        match_centriods.append(col_ind)



                    for k in range(K):
                        latent_fusion_z_common_cluster_points = latent_fusion_z_common[label_pred_zcommon == k]  # common-view 类别为k
                        cluster_variance_common = np.var(latent_fusion_z_common_cluster_points.cpu().detach().numpy())

                        # 3.2 sort the variance on each group pairing cluster on each group
                        for v in range(view_num):
                            single_z_cluster_points = commonZ_term_before[v][label_views[v] == match_centriods[v][k]] #单视图的类别为match_centriods[v][k]
                            cluster_variance_singlez = np.var(single_z_cluster_points.cpu().detach().numpy())
                            if len(latent_fusion_z_common_cluster_points) < len(single_z_cluster_points):
                                copy_time = np.floor(len(single_z_cluster_points)/len(latent_fusion_z_common_cluster_points))
                                tmp_latent_fusion_z_common_cluster_points = np.tile(latent_fusion_z_common_cluster_points.cpu().detach().numpy(), (int(copy_time), 1))
                                tmp_latent_fusion_z_common_cluster_points = torch.as_tensor(tmp_latent_fusion_z_common_cluster_points).to(device)
                                need_sampling_samples = single_z_cluster_points.shape[0] - tmp_latent_fusion_z_common_cluster_points.shape[0]
                                indices = torch.randint(0, latent_fusion_z_common_cluster_points.shape[0], (need_sampling_samples,))
                                new_samples = latent_fusion_z_common_cluster_points[indices]
                                tmp_a = torch.cat([tmp_latent_fusion_z_common_cluster_points, new_samples], dim=0)  # common
                                tmp_b = single_z_cluster_points   # single-view
                            else:
                                copy_time = np.floor(len(latent_fusion_z_common_cluster_points)/len(single_z_cluster_points))
                                tmp_single_z_cluster_points = np.tile(single_z_cluster_points.cpu().detach().numpy(), (int(copy_time), 1))
                                tmp_single_z_cluster_points = torch.as_tensor(tmp_single_z_cluster_points).to(device)
                                need_sampling_samples = latent_fusion_z_common_cluster_points.shape[0] - tmp_single_z_cluster_points.shape[0]
                                indices = torch.randint(0, single_z_cluster_points.shape[0], (need_sampling_samples,))
                                new_samples = torch.as_tensor(single_z_cluster_points[indices])
                                tmp_b = torch.cat([tmp_single_z_cluster_points, new_samples], dim=0) # single-view
                                tmp_a = latent_fusion_z_common_cluster_points   # common

                            # 3.3 KL-divergence of local clusters
                            if cluster_variance_singlez <= cluster_variance_common: # var 值越小越好
                                p_y = F.softmax(tmp_b, dim=-1)  # # reliable cluster;  common view
                                log_unreliable = F.log_softmax(tmp_a, dim=1)  # for unreliable view
                                var_loss += F.kl_div(log_unreliable, p_y, reduction='sum') / 1
                            else:
                                p_y = F.softmax(tmp_a, dim=-1)  # # reliable view;  common view
                                log_unreliable = F.log_softmax(tmp_b, dim=1)  # for unreliable view
                                var_loss += F.kl_div(log_unreliable, p_y, reduction='sum') / 1

                loss = reconstruction_loss + loss_z_norm * config['training']['lambda1'][hyper_lambda1] + \
                       KL_divergence_loss * config['training']['lambda2'][hyper_lambda2] + \
                       var_loss * config['training']['lambda3'][hyper_lambda3]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_z_norm1 += loss_z_norm.item() * config['training']['lambda1'][hyper_lambda1]
                loss_KL_divergence_loss1 += KL_divergence_loss.item() * config['training']['lambda2'][hyper_lambda2] # KL global

            scores = self.evaluation_my_v3(config, logger, train_view, Y_list, device, tmp_idx)
            acc_baseline, nmi_baseline, Precision_baseline, F_measure_baseline, recall_baseline, ARI_baseline, AMI_baseline \
                = scores[0]['kmeans']['accuracy'], scores[0]['kmeans']['NMI'], \
                  scores[0]['kmeans']['precision'], scores[0]['kmeans']['f_measure'], \
                  scores[0]['kmeans']['recall'], scores[0]['kmeans']['ARI'], scores[0]['kmeans']['AMI']

            result_baseline = np.array(
                [acc_baseline, nmi_baseline, Precision_baseline, F_measure_baseline, recall_baseline,
                 ARI_baseline, AMI_baseline]) * 100
            acc_show.extend([acc_baseline])
            nmi_show.extend([nmi_baseline])
            precision_show.extend([Precision_baseline])
            F_measure_show.extend([F_measure_baseline])
            recall_show.extend([recall_baseline])
            ARI_show.extend([ARI_baseline])
            AMI_show.extend([AMI_baseline])


        return scores[0]['kmeans']['accuracy'], scores[0]['kmeans']['NMI'], scores[0]['kmeans']['precision'], \
               scores[0]['kmeans']['f_measure'], scores[0]['kmeans']['recall'], scores[0]['kmeans']['ARI'], \
               scores[0]['kmeans']['AMI'], acc_show, nmi_show, precision_show, F_measure_show, recall_show, ARI_show, AMI_show



    def evaluation_my_v3(self, config, logger, x_train, Y_list, device, tmp_idx):
        # 正确！
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            self.autoencoder3.eval(), self.autoencoder4.eval()
            self.autoencoder5.eval(), self.autoencoder6.eval()

            self.img2txt.eval(), self.txt2img.eval()
            view_num = len(x_train)
            train_view = x_train.copy()

            z = []
            z.append(self.autoencoder1.encoder(train_view[0]))
            z.append(self.autoencoder2.encoder(train_view[1]))
            z.append(self.autoencoder3.encoder(train_view[2]))
            z.append(self.autoencoder4.encoder(train_view[3]))
            z.append(self.autoencoder5.encoder(train_view[4]))
            z.append(self.autoencoder6.encoder(train_view[5]))

            latent_fusion = torch.cat([z[0], z[1], z[2], z[3], z[4], z[5],], dim=0).cpu().numpy()  # 20220509

            scores = evaluation.clustering([latent_fusion], Y_list)
            logger.info("\033[2;29m" + 'view_concat ' + str(scores) + "\033[0m")

            self.autoencoder1.train(), self.autoencoder2.train()  # 观测样本
            self.autoencoder3.train(), self.autoencoder4.train()  # 观测样本
            self.autoencoder5.train(), self.autoencoder6.train()  # 观测样本

            self.img2txt.train(), self.txt2img.train()  # 缺失样本
        return scores, latent_fusion, z, Y_list

