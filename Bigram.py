import numpy
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

names = open("names.txt",'r').read().splitlines()

xs = []
ys = []

a = torch.zeros((27,27),dtype=torch.int32)
chars = sorted(list(set("".join(names))))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in names:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)
        a[idx1][idx2] +=1 

#sns.heatmap(a, annot=True, cmap='coolwarm', fmt="d")
#plt.title("Bigram Heatmap")
#plt.show()

p = a.float()
p = p/p.sum(1,keepdim=True)

print("Statistical Output")
g = torch.Generator().manual_seed(42)
for i in range(10):
    idx = 0
    out = []
    while True:
        p = a[idx].float()
        p = p/p.sum()
        idx = torch.multinomial(p,replacement=True,generator=g,num_samples=1).item()
        if (idx==0) :
            print(''.join(out))
            break
        out.append(itos[idx])

P = a.float()+1
P = P/P.sum(1,keepdim=True)

ll = 0.0
n = 0
for w in names:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        prob = P[idx1][idx2]
        logprob = torch.log(prob)
        ll += logprob
        n += 1

print(f'{ll=}')
nll = -ll
print(f'Average Loss (NLL): {nll/n:.4f}')


xs = torch.tensor(xs)
ys = torch.tensor(ys)
xenc = F.one_hot(xs,num_classes=27).float()
yenc = F.one_hot(ys,num_classes=27).float()

W=torch.randn((27,27),requires_grad=True,generator=g)
for x in range(100):
    logits = (xenc@W) #neural version of the bigram counts matrix
    counts = logits.exp()
    prob = counts/counts.sum(1,keepdim=True)

    loss = -prob[torch.arange(xs.numel()),ys].log().mean()
    W.grad = None
    loss.backward()

    learnrate = 50
    W.data -= learnrate*W.grad #Gradient descent

print()
print("Neural Output")
g = torch.Generator().manual_seed(42)
for i in range(10):
    idx = 0
    nout = []
    while True:
        xenc = F.one_hot(torch.tensor([idx]),num_classes=27).float()
        logits = (xenc@W) #neural version of the bigram counts matrix
        counts = logits.exp()
        prob = counts/counts.sum(1,keepdim=True)
        idx = torch.multinomial(prob,replacement=True,generator=g,num_samples=1).item()
        if (idx==0) :
            print(''.join(nout))
            break
        nout.append(itos[idx])
