import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DeepFM(nn.Module):
    r""" DeepFM model

    Args:
        cat_feature_sizes: List[int], features sizes of categorical features
        n_cont_features: int, number of continuous/numerical features
        embedding_dim: int, dimension of FM second order embedding
        hidden_dims: List[int], dimensions of DNN hidden layers
        dropout: List[float], dropout probability of corresponding hidden layer, should be same length with 'hidden_dims'
        use_cont_for_fm: bool, indicate whether use continuous features in FM part or not

    Shape:
        - Input1(cat_feats): (N, n_cat_features), indicate categorical feature index (LabelEncoding)
        - Input2(cont_feats): (N, n_cont_features), indicate continuous feature value
        - Output: (N, 2), indicate negative prob (idx=0) and positive prob (idx=1)

    Attributes:
        bias: bias for fm part
        fm_cat_first_order_embeddings: MoudleList([Embedding(feature_size, 1)]), for fm first order caterical features
        fm_cont_first_order_embeddings: MoudleList([Embedding(1, 1)]), for fm first order continuous features
        fm_cat_second_order_embeddings: MoudleList([Embedding(feature_size, embedding_dim)]), for fm second order caterical features
        fm_cont_second_order_embeddings: MoudleList([Embedding(1, embedding_dim)]), for fm second order continuous features
        dnn: Sequential, multple dnn hidden layers, including Linear, BatchNorm, Relu, Dropout
        
    Examples::
        >>> batch_size = 10
        >>> cat_feature_sizes = [4,6,8]
        >>> n_cont_feats = 5
        >>> cat_feats = torch.cat([
               torch.randint(0, feature_size, size=(batch_size,1)) for feature_size in cat_feature_sizes
           ],dim=1)
        >>> cont_feats = torch.randn((batch_size,n_cont_feats))
        >>> model = DeepFM(cat_feature_sizes,n_cont_feats,embedding_dim=12,hidden_dims=[50,50],
                   dropout=[0.5,0.5],use_cont_for_fm=True)
        >>> output = model(cat_feats, cont_feats)
        >>> print(output.size())
        torch.Size([10, 2])
    """
    
    def __init__(self, cat_feature_sizes:List[int], n_cont_features:int,
                 embedding_dim:int, hidden_dims:List[int], dropout:List[float],
                 use_cont_for_fm:bool):
        super(DeepFM, self).__init__()
        self.cat_feature_sizes = cat_feature_sizes
        self.n_cat_features = len(self.cat_feature_sizes)
        self.n_cont_features = n_cont_features
        self.use_cont_for_fm = use_cont_for_fm

        """ init fm part """
        self.bias = nn.Parameter(torch.randn(1))

        # first order embedding
        self.fm_cat_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in cat_feature_sizes]
        )
        if use_cont_for_fm:
            self.fm_cont_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(1, 1) for _ in range(n_cont_features)]
            )
        # second order embedding
        self.fm_cat_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, embedding_dim) for feature_size in cat_feature_sizes]
        )
        if use_cont_for_fm:
            self.fm_cont_second_order_embeddings = nn.ModuleList(
                [nn.Embedding(1, embedding_dim) for _ in range(n_cont_features)]
            )

        """ init deep part """
        # self.field_size = len(feature_sizes)
        # input layer + hidden layers + output layer(1)
        if use_cont_for_fm:
            input_dim = (len(cat_feature_sizes) + n_cont_features) * embedding_dim
        else:
            input_dim = len(cat_feature_sizes) * embedding_dim + n_cont_features
        all_dims = [input_dim] + hidden_dims + [1]
        dnn_layers = []
        for i in range(1,len(hidden_dims)+1):
            dnn_layers += [
                nn.Linear(all_dims[i-1],all_dims[i]),
                nn.BatchNorm1d(all_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout[i-1])
            ]
        # append output layer
        dnn_layers.append(nn.Linear(all_dims[-2],all_dims[-1]))
        self.dnn = nn.Sequential(*dnn_layers)

    def get_embedding_arr(self, cat_feats:torch.Tensor, cont_feats:torch.Tensor, order:str):
        assert order in ['first', 'second']
        embeddings = {
            'first_cat': self.fm_cat_first_order_embeddings,
            'second_cat': self.fm_cat_second_order_embeddings,
        }
        if self.use_cont_for_fm:
            embeddings.update({
                'first_cont': self.fm_cont_first_order_embeddings,
                'second_cont': self.fm_cont_second_order_embeddings
            })
        arr = [emb(cat_feats[:, i]) for i, emb in enumerate(embeddings[order+'_cat'])]
        if self.use_cont_for_fm:
            arr += [
                emb(torch.zeros(cont_feats.size(0), dtype=torch.long, device=cont_feats.device)) * cont_feats[:, i:i + 1]
                for i, emb in enumerate(embeddings[order+'_cont'])
            ]
        return arr


    def forward(self, cat_feats:torch.Tensor, cont_feats:torch.Tensor):
        assert cat_feats.size(1) == self.n_cat_features
        assert cont_feats.size(1) == self.n_cont_features

        """ fm first oder """
        # field_size * [N, 1]
        fm_first_order_arr = self.get_embedding_arr(cat_feats,cont_feats,'first')
        # [N]
        fm_first_order = torch.cat(fm_first_order_arr,dim=1).sum(dim=1)

        """ fm second order """
        # field_size * [N, E]
        fm_second_order_arr = self.get_embedding_arr(cat_feats,cont_feats,'second')
        fm_second_order_sum = sum(fm_second_order_arr)
        fm_second_order_sum_square = fm_second_order_sum ** 2
        fm_second_order_square = [emb ** 2 for emb in fm_second_order_arr]
        fm_second_order_square_sum = sum(fm_second_order_square)
        # [N]
        fm_second_order = (0.5 * (
                fm_second_order_sum_square - fm_second_order_square_sum)).sum(
            dim=1)

        """ deep """
        deep_emb = torch.cat(fm_second_order_arr, dim=1)
        if not self.use_cont_for_fm:
            deep_emb = torch.cat([deep_emb, cont_feats], dim=1)
        # [N]
        deep_out = self.dnn(deep_emb).squeeze()

        # summarize
        out_pos = torch.sigmoid( self.bias + fm_first_order + fm_second_order + deep_out).unsqueeze(dim=1)
        out_neg = 1 - out_pos
        # [N, 2]
        output = torch.cat([out_neg,out_pos],dim=1)
        

        return output
