"""Replay the user-confirmed validation DB recording; compare all covered frames and predictions."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import json
import time
import numpy as np
from PIL import Image
from sonar_workbench.sources import DbSeries
from sonar_workbench.inference import FeedingEngine
from sonar_workbench.paths import RESULTS,output_path

db=Path(r'F:\experimental_data\9_12\投食\第5次投食\08.DB')
selected=json.loads((RESULTS/'checkpoints'/'selected_model.json').read_text(encoding='utf-8'))
run=Path(selected['checkpoint']).parent
expected_rows=list(csv.DictReader((run/'val_predictions.csv').open(encoding='utf-8-sig')))
expected={int(r['end_global_frame']):r for r in expected_rows if r['recording_key']=='feeding:5'}
paths=json.loads((RESULTS/'metadata'/'path_lists'/'feeding_005.json').read_text(encoding='utf-8'))
source=DbSeries(db,lambda s:print(s,flush=True))
before={str(s.path):(s.path.stat().st_size,s.path.stat().st_mtime_ns) for s in source.sources}
engine=FeedingEngine(RESULTS/'onnx')
csv_path=output_path('db_replay/verification_recording_5/predictions.csv')
matched=pixel_matches=0
largest=0.0
switches=0
extra_switches=0
started=time.perf_counter()
fields=['global_frame','segment_order','segment_local_frame','source_path','feeding_probability','ui_state',
        'decision','matches_original_validation','processing_ms']
with csv_path.open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader()
    for index in range(source.count):
        begin=time.perf_counter()
        gray,meta=source.read(index)
        if index<len(paths):
            with Image.open(Path(paths[index])) as image:
                original=np.asarray(image.convert('L'),dtype=np.uint8)
            if not np.array_equal(gray,original):
                raise RuntimeError(f'DB/PNG pixel mismatch at global frame {index}')
            pixel_matches+=1
        result=engine.consume(gray,'feeding:5',index,source_frame_number=meta['frame_serial'])
        equivalent=None
        if index in expected:
            row=expected[index]
            difference=abs(result['probability']-float(row['feeding_prob']))
            largest=max(largest,difference)
            equivalent=difference<2e-4 and result['state']==int(row['ui_state'])
            if not equivalent or not result['decision']:
                raise RuntimeError(f'Prediction mismatch at frame {index}: {difference}')
            matched+=1
        switches+=int(result.get('changed',False))
        if index>=len(paths):
            extra_switches+=int(result.get('changed',False))
        writer.writerow(dict(global_frame=index,segment_order=meta['segment_order'],
            segment_local_frame=meta['segment_local_frame'],source_path=meta['source_path'],
            feeding_probability=result['probability'],ui_state=result['state'],decision=result['decision'],
            matches_original_validation=equivalent,processing_ms=(time.perf_counter()-begin)*1000))
        if (index+1)%300==0 or index==source.count-1:
            handle.flush()
            print(f'Replay {index+1}/{source.count}; pixels={pixel_matches}; matched decisions={matched}',flush=True)
source.close()
if matched!=len(expected):
    raise RuntimeError('Not every validation decision was reproduced')
unchanged=all(before[str(s.path)]==(s.path.stat().st_size,s.path.stat().st_mtime_ns) for s in source.sources)
if not unchanged:
    raise RuntimeError('Raw DB metadata changed')
report=dict(passed=True,source=str(db),db_segments=len(source.sources),decoded_frames=source.count,
    training_png_frames_matched=pixel_matches,additional_unlabeled_frames=source.count-len(paths),
    validation_decisions_matched=matched,max_probability_difference=largest,state_switches=switches,
    additional_segment_state_switches=extra_switches,seconds=time.perf_counter()-started,
    raw_db_metadata_unchanged=unchanged,test_used=False,predictions=str(csv_path),
    note='Extra DB tail has no original annotation; no accuracy is claimed for it.')
output_path('reports/db_replay_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

