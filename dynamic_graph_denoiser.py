

import torch
from torch.nn import Linear

class RelationModule(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RelationModule, self).__init__()
        self.lin_in = Linear(embedding_dim*2, hidden_dim)
        self.lin_out = Linear(hidden_dim, 1)

    def forward(self, z1, z2):
        h = self.lin_in(torch.cat((z1, z2), dim=-1))
        h = h.relu()
        h = self.lin_out(h)
        return h.sigmoid()


z1 = torch.Tensor([[1, 1, 1], [2, 2, 2]])
z2 = torch.Tensor([[3, 3, 3], [4, 4, 4]])

relation = RelationModule(3, 2)

print(relation(z1, z2))
