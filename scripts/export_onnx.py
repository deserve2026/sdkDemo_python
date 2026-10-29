"""Export frozen CNN + selected TCN; numerical verification runs in the playback environment."""
import sys
sys.dont_write_bytecode = True
from _entry import PROJECT_ROOT
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.models.cnn_backbone import make_feature_extractor
from src.models.tcn import build_tcn
from src.models.cnn_tcn import CNNTCN
from src.data.dataset import reconstruct_paths
from src.utils.image import preprocess_image
from src.utils.io import load_config, digest, read_csv, write_json, write_text
from src.utils.paths import output_path
from src.utils.experiment import verify_data

class Probabilities(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return torch.softmax(self.model(x), dim=-1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs'/'dataset.yaml')
    args=parser.parse_args()
    config=load_config(args.config)
    verify_data(config)
    selected=json.loads((Path(config['results_dir'])/'checkpoints'/'selected_model.json').read_text(encoding='utf-8'))
    checkpoint=Path(selected['checkpoint'])
    if digest(checkpoint)!=selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    torch.set_num_threads(4)
    cnn=make_feature_extractor(config).eval()
    tcn=build_tcn(payload['model_config']).eval()
    tcn.load_state_dict(payload['model_state_dict'])
    probability_model=Probabilities(tcn).eval()
    combined=Probabilities(CNNTCN(cnn,tcn)).eval()
    output=output_path(config,'onnx/deployment.json').parent
    cases=output/'verification_cases'
    cases.mkdir(exist_ok=True)
    rows=read_csv(Path(config['windows_dir'])/'windows_val.csv')
    chosen=[r for r in rows if r['label']=='normal'][::220][:4]
    positives=[r for r in rows if r['label']=='feeding']
    chosen+=positives[::max(1,len(positives)//5)][:5]
    chosen+=[r for r in rows if r['behavior_source']=='feeding' and r['ignore_reason']=='unlabeled_feeding_source'][::150][:3]
    cache={}
    expected=[]
    with torch.inference_mode():
        for i,row in enumerate(chosen):
            images=np.stack([preprocess_image(p) for p in reconstruct_paths(row,cache)])
            features=cnn(torch.from_numpy(images))
            probability=probability_model(features[None]).numpy()
            path=cases/f'case_{i:02d}.npz'
            np.savez_compressed(path,images=images,features=features.numpy(),probabilities=probability)
            expected.append(dict(file=path.name,window_id=row['window_id'],label=row['label'],sha256=digest(path)))
            print(f'Prepared real-image verification case {i+1}/{len(chosen)}',flush=True)
    # Explicit legacy exporter: this fixed ResNet/Conv1d graph needs onnx, not another torch installation.
    for name,model,example,input_name in [
        ('cnn_encoder.onnx',cnn,torch.zeros(1,3,224,224),'images'),
        ('tcn_classifier.onnx',probability_model,torch.zeros(1,32,512),'features'),
        ('cnn_tcn_feeding.onnx',combined,torch.zeros(1,32,3,224,224),'images')
    ]:
        torch.onnx.export(model,(example,),str(output/name),input_names=[input_name],
                          output_names=['features' if name=='cnn_encoder.onnx' else 'probabilities'],
                          opset_version=17,dynamo=False,external_data=False)
        import onnx
        onnx.checker.check_model(str(output/name))
        print('Exported and checked',name,flush=True)
    deployment=dict(version=1,verified=False,checkpoint_sha256=digest(checkpoint),
        selected_epoch=selected['epoch'],parameters=selected['parameters'],window_size=32,stride=8,
        label_map={'normal':0,'feeding':1},dtype='float32',
        cnn_input_shape=[1,3,224,224],tcn_input_shape=[1,32,512],
        combined_input_shape=[1,32,3,224,224],output_meaning='probabilities [normal, feeding]',
        preprocessing=dict(resize=[224,224],interpolation='Pillow bilinear',color='grayscale repeated to 3 channels',
            scale=255.0,mean=[.485,.456,.406],std=[.229,.224,.225],
            png_orientation='unchanged',db_sdk_orientation='vertical_flip_once',
            validated_db_shape=[512,968],validated_db_pack_type=0),
        model_sha256={name:digest(output/name) for name in ['cnn_encoder.onnx','tcn_classifier.onnx','cnn_tcn_feeding.onnx']},
        verification_cases=expected,test_used=False,fps=None,
        warmup='No classification before 32 consecutive valid frames; reset on seek/gap/new recording',
        decoder_profile_note='DB orientation verified on recording 5; additional conditions require checking')
    write_json(config,'onnx/deployment.json',deployment)
    import yaml
    write_text(config,'onnx/preprocess_config.yaml',yaml.safe_dump(deployment,allow_unicode=True,sort_keys=False))
    print('Export complete; deployment stays disabled until ONNX Runtime verification passes.',flush=True)

if __name__=='__main__':
    main()
