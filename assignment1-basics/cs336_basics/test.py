import torch
#from cs336_basics.model import MultiheadSelfAttention
torch.manual_seed(0)
Q = torch.arange(24).float().reshape(1,3,8)
print('投影后的Q', Q[0,0].tolist())
# mha = MultiheadSelfAttention(d_model=8,num_heads=2)
# x = torch.randn(2,5,8)
# print(mha(x).shape)
