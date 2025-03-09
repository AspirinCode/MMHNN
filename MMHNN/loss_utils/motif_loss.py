import torch
import numpy as np


class Motif_Loss(torch.nn.Module):

    def __init__(self, device, use_cosine_similarity, **kwargs):
        super(Motif_Loss, self).__init__()
        self.device=device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.L2Loss = torch.nn.MSELoss()


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self,out_sub,motif_num):
        loss_list = []
        for i in range(out_sub.size(0)):
            out_sub_i = out_sub[i, :motif_num[i], :]
            similarity_matrix_i = self.similarity_function(out_sub_i, out_sub_i)
            target_similarity_matrix = torch.eye(motif_num[i]).to(self.device)
            loss_i = self.L2Loss(similarity_matrix_i, target_similarity_matrix)
            loss_list.append(loss_i)

        L2_loss = sum(loss_list)/len(loss_list)
        return L2_loss


if __name__ == "__main__":
    import torch

    x = torch.tensor([[[1, 1], [1, 1]],[[1,0],[1,1]]],dtype=torch.float).to(torch.device("cuda:0"))
    print(x)
    motif_num=torch.tensor([2,2]).to(torch.device("cuda:0"))
    print(motif_num)
    Motif_cri=Motif_Loss(device=torch.device("cuda:0"), use_cosine_similarity=True)
    # print(x.sum().view(1))
    loss = Motif_cri(x,motif_num)
    print(loss)