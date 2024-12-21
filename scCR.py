import torch
import copy
from embedder import embedder
from misc.graph_construction import knn_graph

class scCR_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.args.n_nodes = self.adata.obsm["train"].shape[0]
        self.args.n_feat = self.adata.obsm["train"].shape[1]

    def train(self):
        cell_data = torch.Tensor(self.adata.obsm["train"]).to(self.args.device)
        edge_index, edge_weight = knn_graph(cell_data, self.args.k, sym= True)
        self.model = FeaturePropagation(num_iterations=self.args.iter, mask=True, alpha=0.0)
        self.model = self.model.to(self.device)
        denoised_matrix_or = self.model(cell_data, edge_index, edge_weight)
        denoised_matrix_or_c = copy.copy(denoised_matrix_or)
        denoised_matrix_d = torch.cat([denoised_matrix_or_c, -denoised_matrix_or_c], dim=-1)
        means = denoised_matrix_d.mean(dim=0)
        stds = denoised_matrix_d.std(dim=0)
        stds[stds==0]=torch.mean(stds)
        denoised_matrix_d = (denoised_matrix_d-means)/stds
        denoised_matrix_d_T = denoised_matrix_d.T
        b_mask = torch.nonzero(torch.cat([cell_data, -cell_data], dim=-1).T)
        edge_index_c, edge_weight_c = knn_graph(denoised_matrix_d_T,self.args.k_col, sym=True)
        self.model1 = FeaturePropagationT(num_iterations=self.args.iter, mask=True, alpha=0.0)
        denoised_matrix_c = self.model1(denoised_matrix_d_T, edge_index_c, edge_weight_c, b_mask).T
        denoised_matrix_c = denoised_matrix_c * stds + means
        edge_index_new_c, edge_weight_new_c = knn_graph(denoised_matrix_c[:,:denoised_matrix_or.shape[-1]], self.args.k)
        self.model1 = FeaturePropagation(num_iterations=self.args.iter, mask=True, alpha=0.0)
        self.model1 = self.model.to(self.device)
        denoised_matrix_c_cc = self.model1(cell_data, edge_index_new_c, edge_weight_new_c)
        temp = denoised_matrix_or * self.args.alpha + denoised_matrix_c_cc * (1-self.args.alpha)
        edge_index_new, edge_weight_new = knn_graph(temp, self.args.k, sym=self.args.sym)
        self.model2 = FeaturePropagation(num_iterations=self.args.iter, mask=False, alpha=self.args.beta)
        self.model2 = self.model2.to(self.device)
        denoised_matrix = self.model2(cell_data, edge_index_new, edge_weight_new)
        denoised_matrix = denoised_matrix * (1-self.args.gamma) + temp * self.args.gamma
        denoised_matrix = denoised_matrix.detach().cpu().numpy()
        self.adata.obsm['denoised'] = denoised_matrix
        return self.evaluate()

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations, mask, alpha=0.0):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight, ld=None):
        original_x = copy.copy(x)
        nonzero_idx = torch.nonzero(x)
        nonzero_i, nonzero_j = nonzero_idx.t()

        out = x
        n_nodes = x.shape[0]
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
        adj = adj.float()

        if ld is None:
            res = (1-self.alpha) * out
            for _ in range(self.num_iterations):
                out = torch.sparse.mm(adj, out)
                if self.mask:
                    out[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
                else:
                    out.mul_(self.alpha).add_(res)
        else:
            res = (1-self.alpha) * out
            for _ in range(self.num_iterations):
                out = torch.sparse.mm(adj, out)
                if self.mask:
                    out[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
                else:
                    out.mul_(self.alpha).add_(res)
        return out

class FeaturePropagationT(torch.nn.Module):
    def __init__(self, num_iterations, mask, alpha=0.0):
        super(FeaturePropagationT, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight, b_mask):
        original_x = copy.copy(x)
        nonzero_i, nonzero_j = b_mask.t()

        out = x
        n_nodes = x.shape[0]
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
        adj = adj.float()
        for _ in range(self.num_iterations):
            out = torch.mm(adj.to_dense(), out)
            if self.mask:
                out[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
        return out