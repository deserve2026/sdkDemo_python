import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from src.data.feature_dataset import FeatureWindows, quality_valid
from src.models.tcn import build_tcn
from src.training.event_metrics import replay, evaluate_events, selection_rank, window_metrics
from src.training.sampling import sample_epoch
from src.utils.image import preprocess_image


def rows_at(ends, key='feeding:5', label='ignore'):
    return [dict(recording_key=key, behavior_source=key.split(':')[0], end_global_frame=end,
                 start_global_frame=end - 31, window_size=32, label=label,
                 label_id={'ignore': -1, 'normal': 0, 'feeding': 1}[label],
                 ignore_reason='unlabeled_feeding_source' if label == 'ignore' else '',
                 window_id=f'{key}_{end}', split='val') for end in ends]


def event(event_id, start, end):
    return dict(recording_key='feeding:5', event_id=event_id, start_global_frame=start, end_global_frame=end)


PARAMETERS = dict(entry=0.6, exit_threshold=0.4, entry_count=2, exit_count=2)


class ModelTests(unittest.TestCase):
    def test_tcn_shape_causality_and_gradient(self):
        torch.set_num_threads(2)
        model = build_tcn(dict(causal=True, num_classes=2, feature_dim=512, tcn_channels=[8, 8],
                               dilations=[1, 2], kernel_size=3, dropout=0.0))
        x = torch.randn(2, 32, 512)
        changed = x.clone()
        changed[:, 16:] = torch.randn_like(changed[:, 16:]) * 20
        torch.testing.assert_close(model.temporal_features(x)[:, :, :16],
                                   model.temporal_features(changed)[:, :, :16])
        self.assertEqual(tuple(model(x).shape), (2, 2))
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_preprocess_is_fixed_rgb_repeat_and_size(self):
        data = io.BytesIO()
        Image.new('L', (19, 7), 255).save(data, format='PNG')
        data.seek(0)
        result = preprocess_image(data)
        self.assertEqual(result.shape, (3, 224, 224))
        np.testing.assert_allclose(result[:, 0, 0], (1 - np.array([.485, .456, .406])) / np.array([.229, .224, .225]), rtol=1e-6)

    def test_unknown_valid_windows_can_be_predicted_but_corrupt_cannot(self):
        row = rows_at([31])[0]
        self.assertTrue(quality_valid(row))
        dataset = FeatureWindows([row], {'feeding:5': np.zeros((32, 512), dtype=np.float32)})
        self.assertEqual(dataset[0][1], -1)
        row['ignore_reason'] = 'invalid_image'
        with self.assertRaises(ValueError):
            dataset[0]


class SamplingTests(unittest.TestCase):
    def make_rows(self):
        return ([dict(split='train', label='feeding', recording_key='feeding:2') for _ in range(4)] +
                [dict(split='train', label='normal', recording_key=f'normal:{k}') for k in (2, 3) for _ in range(10)] +
                [dict(split='train', label='ignore', recording_key='feeding:2')])

    def test_only_training_all_positive_two_to_one_and_recording_coverage(self):
        rows = self.make_rows()
        indices = sample_epoch(rows, 2, 42)
        self.assertEqual(len(indices), 12)
        self.assertEqual(len(set(indices)), 12)
        self.assertTrue(set(range(4)).issubset(indices))
        self.assertNotIn(24, indices)
        for key in ('normal:2', 'normal:3'):
            self.assertEqual(sum(rows[i]['recording_key'] == key for i in indices), 4)
        self.assertEqual(indices, sample_epoch(rows, 2, 42))
        self.assertNotEqual(indices, sample_epoch(rows, 2, 43))

    def test_rejects_validation_even_if_labels_known(self):
        rows = self.make_rows()
        rows[0]['split'] = 'val'
        with self.assertRaises(ValueError):
            sample_epoch(rows, 2, 42)


class EventTests(unittest.TestCase):
    def test_hysteresis_and_confirmation(self):
        states, episodes = replay(rows_at(range(31, 87, 8)), [.7, .7, .5, .3, .5, .3, .3], **PARAMETERS)
        self.assertEqual(states, [0, 1, 1, 1, 1, 1, 0])
        self.assertEqual(episodes[0]['on_frame'], 39)
        self.assertEqual(episodes[0]['clear_frame'], 79)

    def test_no_backdating_before_decision_end(self):
        result = evaluate_events(rows_at([31, 39, 47]), [.8] * 3, [event(1, 0, 38)], PARAMETERS)
        self.assertEqual(result[0]['detected_events'], 0)

    def test_constant_feeding_is_not_credit_for_multiple_events(self):
        result = evaluate_events(rows_at(range(31, 111, 8)), [.9] * 10,
                                 [event(1, 35, 55), event(2, 70, 95)], PARAMETERS)
        self.assertEqual(result[0]['detected_events'], 1)
        self.assertEqual(result[0]['matched_alarms_not_observed_clearing'], 1)

    def test_repeated_alarm_is_one_hit_plus_repeat(self):
        result = evaluate_events(rows_at(range(31, 79, 8)), [.9, .9, .1, .1, .9, .9],
                                 [event(1, 31, 100)], PARAMETERS)
        self.assertEqual(result[0]['detected_events'], 1)
        self.assertEqual(result[0]['duplicate_event_alarms'], 1)

    def test_invalid_and_recording_boundary_reset_confirmation(self):
        rows = rows_at([31, 39, 47, 55]) + rows_at([31, 39], 'normal:5', 'normal')
        states, episodes = replay(rows, [.9, None, .9, .9, .9, .9], **PARAMETERS)
        self.assertEqual(states, [0, -1, 0, 1, 0, 1])
        self.assertEqual(len(episodes), 2)

    def test_gap_resets(self):
        states, _ = replay(rows_at([31, 47, 55]), [.9] * 3, **PARAMETERS)
        self.assertEqual(states, [0, 0, 1])

    def test_unknown_feeding_intervals_do_not_count_as_normal_false_alarms(self):
        rows = rows_at([31, 39]) + rows_at([31, 39], 'normal:5', 'normal')
        metric, _, _, _, _ = evaluate_events(rows, [.9] * 4, [], PARAMETERS)
        self.assertEqual(metric['normal_false_alarm_episodes'], 1)
        self.assertEqual(metric['normal_valid_decisions'], 2)
        self.assertEqual(metric['normal_feeding_decisions'], 1)

    def test_rank_prefers_recall_then_false_alarms(self):
        rows = rows_at([31, 39]) + rows_at([31, 39], 'normal:5', 'normal')
        events = [event(1, 31, 60)]
        all_high = evaluate_events(rows, [.9] * 4, events, PARAMETERS)[0]
        good = evaluate_events(rows, [.9, .9, .1, .1], events, PARAMETERS)[0]
        missed = evaluate_events(rows, [.1] * 4, events, PARAMETERS)[0]
        self.assertGreater(selection_rank(good, {'f1': 0}), selection_rank(all_high, {'f1': 1}))
        self.assertGreater(selection_rank(all_high, {'f1': 0}), selection_rank(missed, {'f1': 1}))

    def test_window_metrics_exclude_unknown(self):
        rows = rows_at([31], label='feeding') + rows_at([31], 'normal:5', 'normal') + rows_at([47])
        result = window_metrics(rows, [.9, .1, .9], .6)
        self.assertEqual(result['confusion_matrix'], [[1, 0], [0, 1]])
        self.assertEqual(result['support'], 2)


if __name__ == '__main__':
    unittest.main()
