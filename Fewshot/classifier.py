# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import label_to_1hot, dni_linear, LinearDiag, FeatExemplarAvgBlock


class Classifier(nn.Module):
    """
    Classifier whose weights are generated dynamically from synthetic gradient descent:
    Objective: E_{q(w | d_t^l, x_t)}[ -log p(y | feat(x), w) ] + KL( q(w|...) || p(w) )
    Note: we use a simple parameterization
        - q(w | d_t^l, x_t) = Dirac_Delta(w - phi^k),
          phi^k = synthetic_gradient_descent(x_t, phi^0)
          phi^0 = init_net(d_t)
        - p(w) = zero-mean Gaussian and implemented by weight decay
        - p(y=k | feat(x), w) = prototypical network

    :param int nKnovel: number of categories in a task/episode.
    :param int nFeat: feature dimension of the input feature.
    """
    def __init__(self, nKnovel, nFeat):
        super(Classifier, self).__init__()

        self.nKnovel = nKnovel
        self.nFeat = nFeat

        # bias & scale of classifier p(y | x, phi)
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # init_net lambda(d_t^l)
        self.favgblock = FeatExemplarAvgBlock(self.nFeat)
        self.wnLayerFavg = LinearDiag(self.nFeat)

        # grad_net (aka decoupled network interface) phi(x_t)
        # self.dni = dni_linear(self.nKnovel, dni_hidden_size=self.nKnovel*8)

    def apply_classification_weights(self, features, cls_weights):
        """
        Given feature and weights, computing negative log-likelihoods of nKnovel classes
        (B x n x nFeat, B x nKnovel x nFeat) -> B x n x nKnovel

        :param features: features of query set.
        :type features: torch.FloatTensor
        :param cls_weights: generated weights.
        :type cls_weights: torch.FloatTensor
        :return: classification scores
        :rtype: torch.FloatTensor
        """
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)
        cls_scores = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0,
                                                    features, cls_weights.transpose(1,2))
        return cls_scores

    def init_phi(self, features_supp, labels_supp_1hot):
        """
        Compute phi^0 from support set using classwise feature averaging.

        :param features_supp: support features, B x nSupp x nFeat.
        :type features_supp: torch.FloatTensor
        :param labels_supp_1hot: one-hot representation of labels in support set.
        :return: phi^0, B * nKnovel x nFeat
        """
        phi = self.favgblock(features_supp, labels_supp_1hot) # B x nKnovel x nFeat
        batch_size, nKnovel, num_channels = phi.size()
        phi = phi.view(batch_size * nKnovel, num_channels)
        phi = self.wnLayerFavg(phi) # weight each feature differently
        phi = phi.view(-1, nKnovel, num_channels)
        return phi

    def transUpdate(self, features_supp, features, labels_supp, grad_logit, lr=1e-3, retain_graph=True):
        """
        Update phi using synthetic gradient descent from RIMs.

        :param features_supp: support features used for parameter init
        :type features_sup: torch.FloatTensor
        :param features: features to be classified (usually query features)
        :type features: torch.FloatTensor
        :param labels_supp: support labels used for parameter init
        :type labels_supp: torch.FloatTensor
        :param grad_logit: gradient generated for Jacobian-vector product
        :type grad_logit: torch.FloatTensor
        :param lr: learning rate used for gradient descent
        :type lr: float
        :return: phi
        :rtype: torch.FloatTensor
        """
        labels_supp_1hot = label_to_1hot(labels_supp, self.nKnovel)
        phi = self.init_phi(features_supp, labels_supp_1hot)

        batch_size, num_examples = features.size()[:2]
        new_batch_dim = batch_size * num_examples
        cls_scores = self.apply_classification_weights(features, phi)
        cls_scores = cls_scores.view(new_batch_dim, -1) # B * n x nKnovel
        # grad_logit = self.dni(cls_scores) # B * n x nKnovel
        # TODO: directly propagate gradient to variables that generate phi
        grad = torch.autograd.grad([cls_scores], [phi],
                                    grad_outputs=[grad_logit],
                                    create_graph=True, retain_graph=retain_graph,
                                    only_inputs=True)[0] # B x nKnovel x nFeat
        # grad = torch.autograd.grad([cls_scores], [phi],
        #                             grad_outputs=[grad_logit],
        #                             create_graph=True, retain_graph=retain_graph,
        #                             only_inputs=True)[0] # B x nKnovel x nFeat

        # perform synthetic GD
        phi = phi - lr * grad
        # phi.data -= lr * grad

        return phi

    
    def trans_inference(self, features_supp, features, labels_supp, grad_logit, lr=1e-3, retain_graph=True):
        """
        Update phi using synthetic gradient descent from RIMs.

        :param features_supp: support features used for parameter init
        :type features_sup: torch.FloatTensor
        :param features: features to be classified (usually query features)
        :type features: torch.FloatTensor
        :param labels_supp: support labels used for parameter init
        :type labels_supp: torch.FloatTensor
        :param grad_logit: gradient generated for Jacobian-vector product
        :type grad_logit: torch.FloatTensor
        :param lr: learning rate used for gradient descent
        :type lr: float
        :return: phi
        :rtype: torch.FloatTensor
        """
        labels_supp_1hot = label_to_1hot(labels_supp, self.nKnovel)
        phi = self.init_phi(features_supp, labels_supp_1hot)

        batch_size, num_examples = features.size()[:2]
        new_batch_dim = batch_size * num_examples
        cls_scores = self.apply_classification_weights(features, phi)
        cls_scores = cls_scores.view(new_batch_dim, -1) # B * n x nKnovel
        # TODO: directly propagate gradient to variables that generate phi
        grad = torch.autograd.grad([cls_scores], [phi],
                                    grad_outputs=[grad_logit],
                                    create_graph=True, retain_graph=retain_graph,
                                    only_inputs=True)[0] # B x nKnovel x nFeat

        # perform synthetic GD
        phi = phi - lr * grad
        # phi.data -= lr * grad
        cls_scores = self.apply_classification_weights(features, phi)

        return cls_scores

    def get_classification_weights(self, features_supp, labels_supp_1hot):
        """
        Obtain weights for the query set using features_supp, labels_supp and features_query.
        features_supp, labels_supp --> self.init_phi
        features_query --> self.refine_phi

        :features_supp: feat(x_t^l)
        :type features_supp: torch.FloatTensor
        :labels_supp_1hot: one-hot representation of support labels
        :type labels_supp: torch.FloatTensor
        :features_query: feat(x_t)
        :type features_query: torch.FloatTensor
        :lr float: learning rate of synthetic GD
        :return: weights for query set
        :rtype: torch.FloatTensor
        """
        features_supp = F.normalize(features_supp, p=2, dim=features_supp.dim()-1, eps=1e-12)

        weight_novel = self.init_phi(features_supp, labels_supp_1hot)

        return weight_novel


    def forward(self, features_supp, features_query, labels_supp):
        """
        Compute classification scores.
        :labels_supp_1hot: one-hot representation of support labels

        :features_supp: B x nKnovel*nExamplar x nFeat
        :type features_supp: torch.FloatTensor
        :labels_supp: B x nknovel*nExamplar in [0, nKnovel-1]
        :type labels_supp: torch.FloatTensor
        :features_query: B x nKnovel*nTest x nFeat
        :type features_query: torch.FloatTensor
        :return: classification scores
        :rtype: torch.FloatTensor
        """
        labels_supp_1hot = label_to_1hot(labels_supp, self.nKnovel)
        cls_weights = self.get_classification_weights(features_supp, labels_supp_1hot)
        # if features_query == None: # learn using support set
        #     cls_scores = self.apply_classification_weights(features_supp, cls_weights)
        # else:
        cls_scores = self.apply_classification_weights(features_query, cls_weights)

        return cls_scores


if __name__ == "__main__":
    net = Classifier(nKnovel=5, nFeat=512)
    net = net.cuda()

    features_supp = torch.rand((8, 5 * 1, 512)).cuda()
    features_query = torch.rand((8, 5 * 15, 512)).cuda()
    labels_supp = torch.randint(5, (8, 5 * 1)).cuda()
    lr = 1e-3

    cls_scores = net(features_supp, labels_supp, features_query, lr)
    print(cls_scores.size())

