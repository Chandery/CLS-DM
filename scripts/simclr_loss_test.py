import torch

batch_size = 64
pos = torch.rand(batch_size, 1)
neg = torch.rand(batch_size, batch_size)

# InfoNCE loss
cri = torch.nn.CrossEntropyLoss()
logits = torch.cat([pos, neg], dim=1)
labels = torch.zeros(batch_size).long()
loss = cri(logits, labels)
print(loss)

# HCL loss
pos_exp = torch.exp(pos)
neg_exp = torch.exp(neg).sum(dim=1)

hcl_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
print(hcl_loss)