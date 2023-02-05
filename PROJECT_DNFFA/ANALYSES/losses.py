import torch
from torch import nn

def get_loss_fn(args):
    
    if args.sparse_pos is True:
        criterion = SparsePositiveCrossEntropyLoss(args.l1_pos_lambda,
                                                   args.l1_neg_lambda)   
    else:
        criterion = nn.CrossEntropyLoss()
                
    return criterion
        
class SparsePositiveCrossEntropyLoss():
    
    # by default, sparse positivity is off (lambdas = 0)
    def __init__(self, l1_pos_lambda=0, l1_neg_lambda=0):
        self.l1_pos_lambda = l1_pos_lambda
        self.l1_neg_lambda = l1_neg_lambda
        
    def compute_loss(self, weights, out, labs):
        
        loss_fn = nn.CrossEntropyLoss()

        clf_loss = loss_fn(out, labs)

        l1_pos_loss = self.l1_pos_lambda * torch.sum(torch.abs(weights[weights>0]))
        l1_neg_loss = self.l1_neg_lambda * torch.sum(torch.abs(weights[weights<0]))
        
        total_loss = clf_loss + l1_pos_loss + l1_neg_loss
    
        return total_loss, clf_loss, l1_pos_loss, l1_neg_loss
    