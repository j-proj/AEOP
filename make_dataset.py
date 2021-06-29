import numpy
from datasets import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


AWA_IMG_PATH = 'C:/Users/jdlear/Documents/AEOP/Animals_with_Attributes2/JPEGImages'
DATA_SAVE_PATH = 'awa_huggingface_dataset/awa2'


transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224)])

data_set2 = ImageFolder(AWA_IMG_PATH, transform)

data_dict = dict()
data_dict['img'] = []
data_dict['label'] = []
for img, label in data_set2:
    data_dict['img'].append(numpy.array(img))
    data_dict['label'].append(label)

dataset = Dataset.from_dict(data_dict)

dataset.save_to_disk('awa_huggingface_dataset/awa2')