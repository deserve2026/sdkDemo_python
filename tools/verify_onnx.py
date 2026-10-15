"""Numerical equivalence only. Does not tune a model or evaluate the held-out test set."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import onnxruntime as ort
from sonar_workbench.paths import RESULTS, output_path, sha256

root=RESULTS/'onnx'
deployment=json.loads((root/'deployment.json').read_text(encoding='utf-8'))
options=ort.SessionOptions()
options.intra_op_num_threads=4
options.inter_op_num_threads=1
options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
sessions={}
for name,expected in deployment['model_sha256'].items():
    if sha256(root/name)!=expected:
        raise ValueError('Model checksum mismatch: '+name)
    sessions[name]=ort.InferenceSession(str(root/name),options,providers=['CPUExecutionProvider'])
records=[]
started=time.perf_counter()
for case in deployment['verification_cases']:
    path=root/'verification_cases'/case['file']
    if sha256(path)!=case['sha256']:
        raise ValueError('Verification case changed')
    with np.load(path,allow_pickle=False) as data:
        images,features,expected=data['images'],data['features'],data['probabilities']
        actual_features=np.concatenate([sessions['cnn_encoder.onnx'].run(None,{'images':im[None]})[0] for im in images],axis=0)
        actual=sessions['tcn_classifier.onnx'].run(None,{'features':actual_features[None]})[0]
        tcn_only=sessions['tcn_classifier.onnx'].run(None,{'features':features[None]})[0]
        combined=sessions['cnn_tcn_feeding.onnx'].run(None,{'images':images[None]})[0]
        record=dict(window_id=case['window_id'],label=case['label'],
                    feature_max_abs_error=float(np.max(np.abs(actual_features-features))),
                    split_probability_max_abs_error=float(np.max(np.abs(actual-expected))),
                    tcn_probability_max_abs_error=float(np.max(np.abs(tcn_only-expected))),
                    combined_probability_max_abs_error=float(np.max(np.abs(combined-expected))))
        np.testing.assert_allclose(actual_features,features,atol=2e-4,rtol=2e-4)
        for probabilities in (actual,tcn_only,combined):
            np.testing.assert_allclose(probabilities,expected,atol=2e-4,rtol=2e-4)
        records.append(record)
        print(record,flush=True)
report=dict(passed=True,onnxruntime=ort.__version__,cases=records,seconds=time.perf_counter()-started,
            checkpoint_sha256=deployment['checkpoint_sha256'],test_used=False)
output_path('reports/onnx_export_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
deployment['verified']=True
deployment['verification_report']=str(RESULTS/'reports'/'onnx_export_verification.json')
(root/'deployment.json').write_text(json.dumps(deployment,ensure_ascii=False,indent=2),encoding='utf-8')
yaml=root/'preprocess_config.yaml'
yaml.write_text(yaml.read_text(encoding='utf-8').replace('\nverified: false\n','\nverified: true\n',1),encoding='utf-8')
print('All ONNX comparisons passed. Deployment enabled.',flush=True)

