import torch
from vit import ViT
from dl import LoadData
from torch.utils.data	import DataLoader	

from	pathlib	import	Path
import torch.nn.functional as F
import torch.nn as nn
from lenet import LeNet5

class VitConfig:
  def __init__(self, dim, depth, heads, mlp_dim):
    self.dim = dim
    self.depth = depth
    self.heads = heads
    self.mlp_dim = mlp_dim

class MLP(nn.Module):
  def __init__(self, feat_num, hidden_dim, cls_num, region_num):
    super().__init__()
    self.feat_num = feat_num
    self.hidden_dim = hidden_dim
    self.cls_num = cls_num
    self.region_num = region_num
    self.fc1 = nn.Linear(feat_num, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, cls_num)
    self.fc3 = nn.Linear(hidden_dim, region_num)
    self.dropout = nn.Dropout(p=0.3)

  def forward(self, img):
    x = img.view(-1, self.feat_num)
    x = self.fc1(x)
    y = self.dropout(x)
    y = F.sigmoid(y)
    y_cls = self.fc2(y)
    y_region = self.fc3(y)
    return y_cls, y_region

if __name__ == "__main__":
  
  # config = {"dim": 256, "depth":6, "heads": 16, "mlp_dim": 512}
  # config = VitConfig(256, 6, 16, 512)
  # config = VitConfig(32, 4, 4, 128)
  
  # mod = MLP(feat_num = 32*32*5, hidden_dim=128, cls_num=4, region_num=20)
  mod = LeNet5(4, 20)
  # mod = ViT(
  #   image_size = 32,
  #   patch_size = 8,
  #   num_classes = 4,
  #   dim = config.dim,
  #   depth = config.depth,
  #   heads = config.heads,
  #   mlp_dim = config.mlp_dim,
  #   channels = 5,
  #   dropout = 0.1,
  #   emb_dropout = 0.1
  # )

  mod = mod.cuda()
  # optimizer = torch.optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
  optimizer = torch.optim.AdamW(mod.parameters(), lr=1e-4, weight_decay=1e-2)
  # optimizer = torch.optim.Adam(mod.parameters(), lr=3e-5)
  

  root_path	=	Path("/data/path/contain/data_dir")
  data_dir	=	root_path.joinpath("testdata_ata")
  label_path	=	data_dir.joinpath("ata_score.csv")

  dataset	=	LoadData(img_dir=data_dir,	label_path=label_path)

  # split train & val
  train_size = int(0.85 * len(dataset))
  valid_size = len(dataset) - train_size
  print(f"[Train]: {train_size} [Val]: {valid_size}")
  
  # torch.manual_seed(2024)
  torch.manual_seed(3407)
  train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
  
  BATCH_SIZE = 64
  train_loader	=	DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader	=	DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(1.0 / dataset.dist).cuda()) 
  loss_fn_l = torch.nn.CrossEntropyLoss() 
  loss_fn_r = torch.nn.CrossEntropyLoss() 
  loss_fn_i = torch.nn.MSELoss()

  losses = []
  accs = []
  for epo in range(1000):
    # train
    running_loss = 0.
    last_loss = 0.

    running_acc = 0.
    last_acc = 0.

    mod.train(True)
    for i, data in enumerate(train_loader):
      img, label_cpu, region_cpu = data
      # print(img.size())
      img = img.cuda()
      label = label_cpu.cuda()
      region = region_cpu.cuda()
      optimizer.zero_grad()

      preds_label, preds_region, preds_img = mod(img)

      loss_label = loss_fn_l(preds_label, label)
      loss_region = loss_fn_r(preds_region, region)
      loss_img = loss_fn_i(preds_img, img)
      # print(f"{loss_label} {loss_region} {loss_img}")
      loss = loss_label + 0.3*loss_region + 0.2*loss_img
      loss.backward()

      optimizer.step()
      
      preds_label_cpu = preds_label.cpu()
      acc = (torch.argmax(preds_label_cpu, dim=1) == label_cpu).sum()

      running_loss += loss.item() 
      running_acc += acc / len(label)

      if i % 10 == 9:
          last_loss = running_loss / 10 # loss per batch
          last_acc = running_acc / 10
          print(f"Epoch {epo}  Batch {i + 1} Loss: {last_loss} Acc: {last_acc : .2%}")
          running_loss = 0.
          running_acc = 0.

    losses.append(last_loss)
    accs.append(last_acc)
    # eval
    mod.eval()
    ncorrect = 0
    ncorrect_per_cls = [0, 0, 0, 0]
    n_per_cls = [0, 0, 0, 0]
    with torch.no_grad():
      for i, vdata in enumerate(valid_loader):
        vinputs, vlabels, vregion = vdata
        vinputs = vinputs.cuda()
        
        voutputs, _, _ = mod(vinputs)
        voutputs = torch.argmax(voutputs.cpu(), dim=1)

        ncorrect += torch.sum(vlabels == voutputs)
        for i in range(len(ncorrect_per_cls)):
          ncorrect_per_cls[i] += torch.sum((vlabels == voutputs) & (vlabels == i))
          n_per_cls[i] += torch.sum(vlabels == i)

        # print(f"Pred: {voutputs} GroundTruth: {vlabels}")
        # print(f"Num correct: {ncorrect}")
    print(f"Accuracy: {ncorrect/valid_size : .2%}[{ncorrect}/{valid_size}] {ncorrect_per_cls[0]/valid_size:.2%} {ncorrect_per_cls[1]/valid_size:.2%} {ncorrect_per_cls[2]/valid_size:.2%} {ncorrect_per_cls[3]/valid_size:.2%}")
    print(f"Accuracy: {ncorrect/valid_size : .2%}[{ncorrect}/{valid_size}] {n_per_cls[0]/valid_size:.2%} {n_per_cls[1]/valid_size:.2%} {n_per_cls[2]/valid_size:.2%} {n_per_cls[3]/valid_size:.2%}")
  print(losses)
  print(accs)

