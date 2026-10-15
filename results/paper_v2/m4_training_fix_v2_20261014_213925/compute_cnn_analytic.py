"""Shape-based DeepCrack convolution MAC count at the locked 256-square input."""
import json
from pathlib import Path
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'DeepSegmentor'))
import torch
from models.deepcrack_networks import DeepCrackNet

if __name__=='__main__':
    start=time.perf_counter()
    torch.set_num_threads(1)
    model=DeepCrackNet(in_nc=3,num_classes=1,ngf=64,norm='batch').eval()
    rows=[];handles=[]
    def register(name,module):
        def count(layer,inputs,output):
            per_output=(layer.in_channels//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1]
            rows.append(dict(layer=name,output_shape=list(output.shape),macs=int(output.numel()*per_output)))
        handles.append(module.register_forward_hook(count))
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):register(name,module)
        if isinstance(module,torch.nn.ConvTranspose2d):
            raise RuntimeError('Add a separate transpose-convolution counting rule')
    with torch.no_grad():model(torch.zeros(1,3,256,256))
    for h in handles:h.remove()
    result=dict(model='DeepCrack',input=[1,3,256,256],parameters=sum(p.numel() for p in model.parameters()),
        gflops_analytic=2*sum(row['macs'] for row in rows)/1e9,convention='2 x convolution MACs / 1e9',
        scope='All Conv2d layers including side outputs and fusion; excludes normalization, activations, interpolation and CPU morphology. No dataset images or checkpoint values required.',
        layers=rows,wall_seconds=time.perf_counter()-start)
    with (OUT/'cnn_analytic.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2))
