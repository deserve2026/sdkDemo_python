import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.data.audit import event_is_placeholder, inspect_png, parse_folder, parse_frame
from src.data.events import map_event, normalize_event
from src.data.splitting import validate_split_map
from src.data.timeline import build_timeline
from src.data.windowing import classify_window, feeding_mask


def frame(rid, segment, local, valid=1):
    folder = f'gts_feed_{rid}' + (f'_{segment}' if segment else '')
    return {'behavior': 'feeding', 'recording_id': rid, 'recording_key': f'feeding:{rid}',
            'segment_order': segment, 'folder_name': folder, 'local_frame': local,
            'is_valid': valid, 'image_path': str(Path('read_only_data') / folder / f'frame_{local:06d}.png')}


class NamingTests(unittest.TestCase):
    def test_segment_zero_and_numeric_segment(self):
        self.assertEqual(parse_folder('gts_feed_1'), ('feeding', 1, 0))
        self.assertEqual(parse_folder('gts_feed_1_10'), ('feeding', 1, 10))
        self.assertEqual(parse_folder('gts_normal_13_2'), ('normal', 13, 2))

    def test_invalid_folder(self):
        with self.assertRaises(ValueError):
            parse_folder('gts_feed_unknown')

    def test_frame_formats(self):
        for value in (289, '289', 'frame_000289', 'frame_000289.png'):
            self.assertEqual(parse_frame(value), 289)
        with self.assertRaises(ValueError):
            parse_frame('camera_20261002_289.png')

    def test_species_only_placeholder(self):
        self.assertTrue(event_is_placeholder({'species': 'gts（高体鰤）'}))
        self.assertFalse(event_is_placeholder({'species': 'gts', 'event_id': '1'}))

    def test_partial_event_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_event({'event_id': '1'})


class TimelineTests(unittest.TestCase):
    def test_actual_lengths_and_recording_isolation(self):
        rows = [frame(1, 0, i) for i in range(3)] + [frame(1, 1, i) for i in range(2)] + [frame(2, 0, 0)]
        result = build_timeline(list(reversed(rows)))
        first = [r for r in result if r['recording_id'] == 1]
        self.assertEqual([r['global_frame'] for r in first], [0, 1, 2, 3, 4])
        self.assertEqual(first[3]['local_frame'], 0)
        self.assertEqual(first[3]['continuity_id'], first[2]['continuity_id'])
        self.assertEqual(result[-1]['global_frame'], 0)

    def test_gap_breaks_continuity_without_compressing_physical_claim(self):
        rows = build_timeline([frame(1, 0, 0), frame(1, 0, 2)])
        self.assertEqual([r['global_frame'] for r in rows], [0, 1])
        self.assertNotEqual(rows[0]['continuity_id'], rows[1]['continuity_id'])

    def test_bad_image_breaks_both_sides(self):
        rows = build_timeline([frame(1, 0, 0), frame(1, 0, 1, 0), frame(1, 0, 2)])
        self.assertEqual(len({r['continuity_id'] for r in rows}), 3)

    def test_missing_segment_breaks(self):
        rows = build_timeline([frame(1, 0, 0), frame(1, 2, 0)])
        self.assertNotEqual(rows[0]['continuity_id'], rows[1]['continuity_id'])

    def test_duplicate_frame_is_fatal(self):
        with self.assertRaises(ValueError):
            build_timeline([frame(1, 0, 0), frame(1, 0, 0)])

    def test_cross_folder_event_uses_actual_indices(self):
        rows = build_timeline([frame(1, 0, i) for i in range(3)] + [frame(1, 1, i) for i in range(2)])
        event = normalize_event({'recording_id': '1', 'event_id': '1', 'start_folder': 'gts_feed_1',
                                 'start_frame': 'frame_000002', 'end_folder': 'gts_feed_1_1', 'end_frame': 'frame_000001.png'})
        mapped = map_event(event, {(r['folder_name'], r['local_frame']): r for r in rows}, {'feeding:1': rows})
        self.assertEqual((mapped['start_global_frame'], mapped['end_global_frame']), (2, 4))
        self.assertEqual(mapped['event_frames_present'], 3)

    def test_cross_recording_event_is_rejected(self):
        rows = build_timeline([frame(1, 0, 0), frame(2, 0, 0)])
        event = normalize_event({'recording_id': '1', 'event_id': '1', 'start_folder': 'gts_feed_1',
                                 'start_frame': '0', 'end_folder': 'gts_feed_2', 'end_frame': '0'})
        with self.assertRaises(ValueError):
            map_event(event, {(r['folder_name'], r['local_frame']): r for r in rows}, {})


class LabelTests(unittest.TestCase):
    def test_threshold_25_versus_26(self):
        self.assertEqual(classify_window('feeding', 25, 32, .8)[0], 'ignore')
        self.assertEqual(classify_window('feeding', 26, 32, .8)[0], 'feeding')

    def test_unlabeled_feeding_is_never_normal(self):
        self.assertEqual(classify_window('feeding', 0, 32, .8), ('ignore', 'unlabeled_feeding_source'))
        self.assertEqual(classify_window('normal', 0, 32, .8)[0], 'normal')

    def test_quality_overrides_positive_label(self):
        self.assertEqual(classify_window('feeding', 32, 32, .8, valid=False)[0], 'ignore')
        self.assertEqual(classify_window('feeding', 32, 32, .8, continuous=False)[0], 'ignore')
        self.assertEqual(classify_window('normal', 0, 32, .8, continuous=False)[0], 'ignore')

    def test_overlap_is_union_not_sum(self):
        self.assertEqual(feeding_mask(6, [(1, 3), (2, 4)]), [0, 1, 1, 1, 1, 0])

    def test_out_of_bounds_event_is_not_clipped(self):
        with self.assertRaises(ValueError):
            feeding_mask(6, [(0, 6)])


class SplitTests(unittest.TestCase):
    def test_source_namespace_is_preserved(self):
        mapping = {'feeding': {'train': [1], 'val': [], 'test': []}, 'normal': {'train': [], 'val': [1], 'test': []}}
        result = validate_split_map(mapping, {('feeding', 1), ('normal', 1)})
        self.assertEqual(result[('feeding', 1)], 'train')
        self.assertEqual(result[('normal', 1)], 'val')

    def test_recording_cannot_cross_splits(self):
        with self.assertRaises(ValueError):
            validate_split_map({'feeding': {'train': [1], 'val': [1], 'test': []}}, {('feeding', 1)})

    def test_extra_recordings_are_not_randomly_assigned(self):
        with self.assertRaises(ValueError):
            validate_split_map({'feeding': {'train': [1], 'val': [], 'test': []}}, {('feeding', 1), ('feeding', 2)})


class ImageAuditTests(unittest.TestCase):
    def test_full_png_validation(self):
        buffer = io.BytesIO()
        Image.new('L', (12, 8), color=100).save(buffer, format='PNG')
        stat = SimpleNamespace(st_size=len(buffer.getvalue()), st_mtime_ns=1)
        with patch.object(Path, 'read_bytes', return_value=buffer.getvalue()), patch.object(Path, 'stat', return_value=stat):
            result = inspect_png(Path('中文（路径）') / 'frame_000001.png')
        self.assertEqual((result['is_valid'], result['image_width'], result['image_height']), (1, 12, 8))

    def test_corrupt_png_is_reported(self):
        stat = SimpleNamespace(st_size=7, st_mtime_ns=1)
        with patch.object(Path, 'read_bytes', return_value=b'not png'), patch.object(Path, 'stat', return_value=stat):
            result = inspect_png(Path('frame_000001.png'))
        self.assertEqual(result['is_valid'], 0)
        self.assertTrue(result['issues'])


if __name__ == '__main__':
    unittest.main()
