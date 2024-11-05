from	torch.utils.data	import	DataLoader	
from	torch.utils.data	import	dataset

from	pathlib	import	Path
import	nibabel	as	nib
import	csv

import numpy as np

class	LoadData(dataset.Dataset):
  def	__init__(self,	img_dir,	label_path)	->	None:
    super().__init__()
    self.img_dir = img_dir
    self.label_path	=	label_path
    self.get_all_imgs_labels()
    self.get_mean_std()
    self.get_class_distribution()

  def	__getitem__(self,	index):
    #	return	super().__getitem__(index)
    res	=	self.all_imgs_labels[index]
    img_arr	=	nib.load(res[0]).get_fdata().astype(np.float32)
    label	=	int(res[1][0])
    region = res[2]
    return	((img_arr - self.mean) / self.std).transpose(2, 0, 1), label, region
  
  def	__len__(self):
    return	len(self.all_imgs_labels)

  def	get_all_imgs_labels(self):
    self.all_imgs_labels	=	[]
    img_path	=	Path(self.img_dir)
    with open(self.label_path, 'r', encoding='utf-8')	as file:
      reader = csv.reader(file)
      for	row	in reader:
        pat	=	img_path.joinpath(row[0])
        if not pat.is_dir(): continue
        for i in range(1,21):
          reg_name = 'region_'+str(i)	+	".nii"
          reg_file = pat.joinpath(reg_name)
          if not reg_file.exists():	continue
          self.all_imgs_labels.append([reg_file,row[i],i-1])

  def get_mean_std(self):
    num_channel = 5
    mean_g = np.array([0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float32)
    std_g = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for img_path, _, _ in self.all_imgs_labels:
      img_arr	=	nib.load(img_path).get_fdata().astype(np.float32) # (32, 32, 5)
      for ch in range(num_channel): 
        mean_g[ch] += np.mean(img_arr[:, :, ch])
        std_g[ch] += np.std(img_arr[:, :, ch])
    mean_g = mean_g / len(self.all_imgs_labels) 
    std_g = std_g / len(self.all_imgs_labels) 
    self.mean = mean_g
    self.std = std_g 

  def get_class_distribution(self):
    num_cls = 4
    dist = [0, 0, 0, 0]
    for _, cls, _ in self.all_imgs_labels:
      dist[int(cls[0])] += 1
    print(dist)
    self.dist = np.array(dist, dtype=np.float32) / len(self.all_imgs_labels)
      
      

  

if	__name__	==	"__main__":
  #	print(torch.__version__)
  root_path	=	Path("/data/framework/wwz/project/why/")
  ################
  data_dir	=	root_path.joinpath("testdata_ata")
  label_path	=	data_dir.joinpath("ata_score.csv")
  #################
  train_dataset	=	LoadData(img_dir=data_dir,	label_path=label_path)
  train_loader	=	DataLoader(
  dataset=train_dataset,
  batch_size=2,
  shuffle=True
  )
  
  for image, label in train_loader:
    print(image.shape)
    print(label)
