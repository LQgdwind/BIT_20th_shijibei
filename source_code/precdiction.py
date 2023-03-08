import torch
import pandas as pd
import torchvision
from torch.utils import data
import os
from source_code.trainBatch import try_all_gpus
import source_code.OurModel as OurModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

net = OurModel.model()
preds = []
test_ds = torchvision.datasets.ImageFolder("..//testdata",torchvision.transforms.Compose([torchvision.transforms.Resize([224,224]),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.5420377, 0.5206433, 0.49089015],[0.21957076, 0.21739748, 0.2220452])]))
test_iter = data.DataLoader(test_ds, batch_size=8, shuffle=False, drop_last=False)

def precdict(net = OurModel.model() ,devices = try_all_gpus()):
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load("..//model_state//rubbish_classification.pkl"))
    net.eval()
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: test_ds.classes[x])
    df.to_csv('..//result.csv', index=False)

devices = try_all_gpus()
precdict(net, devices)
