import io
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from sonar_workbench.inference import preprocess, Debouncer, FeedingEngine

class Encoder:
    def run(self, _, inputs):
        return [np.ones((1,512),dtype=np.float32)]
class Temporal:
    def run(self, _, inputs):
        return [np.array([[.1,.9]],dtype=np.float32)]
def engine():
    e=FeedingEngine.__new__(FeedingEngine)
    e.encoder,e.temporal=Encoder(),Temporal()
    e.features=deque(maxlen=32)
    e.debounce=Debouncer()
    e.reset()
    return e

class InferenceTests(unittest.TestCase):
    def test_preprocess_matches_training(self):
        g=np.full((512,968),200,dtype=np.uint8)
        value=preprocess(g)
        self.assertEqual(value.shape,(1,3,224,224))
        expected=(200/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
        np.testing.assert_allclose(value[0,:,0,0],expected,rtol=1e-6)
    def test_first_decision_31_next_39_then_debounced_feeding(self):
        e=engine(); g=np.zeros((4,4),dtype=np.uint8); decisions=[]
        for frame in range(48):
            r=e.consume(g,'record',frame)
            if r['decision']: decisions.append((frame,r['state']))
        self.assertEqual(decisions,[(31,0),(39,1),(47,1)])
    def test_seek_new_recording_invalid_reset(self):
        e=engine(); g=np.zeros((4,4),dtype=np.uint8)
        for frame in range(40): e.consume(g,'record',frame)
        self.assertTrue(e.debounce.active)
        r=e.consume(g,'record',100)
        self.assertIsNone(r['state'])
        self.assertEqual(r['buffer_size'],1)
        r=e.consume(g,'other',101)
        self.assertEqual(r['buffer_size'],1)
        e.consume(None,'other',102)
        self.assertEqual(len(e.features),0)
    def test_device_gap_resets_even_with_consecutive_array_indices(self):
        e=engine(); g=np.zeros((4,4),dtype=np.uint8)
        for i in range(40):
            e.consume(g,'db',i,source_frame_number=(2**32-20+i)%(2**32))
        self.assertTrue(e.debounce.active)  # Counter wrap is continuous.
        result=e.consume(g,'db',40,source_frame_number=30)  # Device frames 20..29 missing.
        self.assertIsNone(result['state'])
        self.assertEqual(result['buffer_size'],1)
    def test_hysteresis(self):
        d=Debouncer()
        actual=[d.update(p)[0] for p in [.9,.9,.7,.5,.7,.5,.5]]
        self.assertEqual(actual,[0,1,1,1,1,1,0])
    def test_exported_models_are_required_to_be_verified(self):
        with patch('pathlib.Path.read_text',return_value='{\"verified\": false}'):
            with self.assertRaises(ValueError):
                FeedingEngine(Path('unused'))

if __name__=='__main__': unittest.main()

