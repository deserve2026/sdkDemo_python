"""Causal UI replay, one-alarm/one-event matching, no unknown-as-normal shortcut."""
from collections import defaultdict
import math


def replay(rows, probabilities, entry, exit_threshold, entry_count=2, exit_count=2, stride=8):
    if not 0 <= exit_threshold < entry <= 1 or min(entry_count, exit_count, stride) < 1:
        raise ValueError('Invalid debounce configuration')
    if len(rows) != len(probabilities):
        raise ValueError('Prediction length mismatch')
    states, episodes = [], []
    active, high, low, last_key, last_end, current = False, 0, 0, None, None, None

    def close(reason, clear_frame=None):
        nonlocal current
        if current is not None:
            current.update(close_reason=reason, clear_frame=clear_frame)
            episodes.append(current)
            current = None

    for row, prob in zip(rows, probabilities):
        key, end = row['recording_key'], int(row['end_global_frame'])
        if key == last_key and last_end is not None and end <= last_end:
            raise ValueError('Replay must be chronological within each recording')
        if key != last_key or last_end is None or end - last_end != stride:
            close('recording_end' if key != last_key else 'gap')
            active, high, low = False, 0, 0
        last_key, last_end = key, end
        if prob is None or not math.isfinite(float(prob)):
            close('invalid_window')
            active, high, low = False, 0, 0
            states.append(-1)
            continue
        if not active:
            high = high + 1 if prob >= entry else 0
            if high >= entry_count:
                active, high, low = True, 0, 0
                current = dict(recording_key=key, behavior_source=row['behavior_source'],
                               on_frame=end, decision_frames=[])
        else:
            low = low + 1 if prob <= exit_threshold else 0
            if low >= exit_count:
                close('cleared', end)
                active, high, low = False, 0, 0
        if active:
            current['decision_frames'].append(end)
        states.append(int(active))
    close('recording_end')
    return states, episodes


def window_metrics(rows, probabilities, threshold):
    tn = fp = fn = tp = 0
    for row, p in zip(rows, probabilities):
        if row['label'] == 'ignore' or p is None:
            continue
        y, pred = int(row['label_id']), int(p >= threshold)
        tn += int(y == 0 and pred == 0)
        fp += int(y == 0 and pred == 1)
        fn += int(y == 1 and pred == 0)
        tp += int(y == 1 and pred == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                accuracy=(tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
                confusion_matrix=[[tn, fp], [fn, tp]], support=tp + tn + fp + fn)


def evaluate_events(rows, probabilities, events, parameters, stride=8):
    states, episodes = replay(rows, probabilities, stride=stride, **parameters)
    groups = defaultdict(list)
    for event in events:
        groups[event['recording_key']].append(event)
    matched, event_overlaps = {}, defaultdict(list)
    for episode_index, episode in enumerate(episodes):
        own_events = sorted(groups[episode['recording_key']], key=lambda e: int(e['start_global_frame']))
        candidates = []
        for event in own_events:
            token = (event['recording_key'], str(event['event_id']))
            inside = [t for t in episode['decision_frames']
                      if int(event['start_global_frame']) <= t <= int(event['end_global_frame'])]
            if inside:
                event_overlaps[token].append(episode_index)
                if token not in matched:
                    candidates.append((token, inside[0]))
        # A continuously active alarm can receive credit for only one event.
        if candidates:
            token, decision = candidates[0]
            matched[token] = (episode_index, decision)
    details, delays, clear_delays = [], [], []
    uncleared = 0
    for event in events:
        key = (event['recording_key'], str(event['event_id']))
        match = matched.get(key)
        delay = clear_delay = None
        if match:
            index, decision = match
            delay = decision - int(event['start_global_frame'])
            delays.append(delay)
            clear = episodes[index]['clear_frame']
            if clear is None:
                uncleared += 1
            else:
                clear_delay = max(0, clear - int(event['end_global_frame']))
                clear_delays.append(clear_delay)
        details.append(dict(recording_key=key[0], event_id=key[1], detected=bool(match),
                            start_global_frame=int(event['start_global_frame']),
                            end_global_frame=int(event['end_global_frame']),
                            decision_frame=match[1] if match else None,
                            onset_delay_frames=delay, post_event_clear_delay_frames=clear_delay,
                            overlapping_alarm_episodes=len(event_overlaps[key]),
                            extra_alarm_episodes=max(0, len(event_overlaps[key]) - 1)))
    normal_count = sum(r['behavior_source'] == 'normal' and state >= 0 for r, state in zip(rows, states))
    normal_positive = sum(r['behavior_source'] == 'normal' and state == 1 for r, state in zip(rows, states))
    normal_episodes = [e for e in episodes if e['behavior_source'] == 'normal']
    per_record = []
    for key in sorted({r['recording_key'] for r in rows}):
        pairs = [(r, s) for r, s in zip(rows, states) if r['recording_key'] == key]
        own = [e for e in episodes if e['recording_key'] == key]
        per_record.append(dict(recording_key=key, behavior_source=pairs[0][0]['behavior_source'],
                               valid_decisions=sum(s >= 0 for _, s in pairs),
                               feeding_decisions=sum(s == 1 for _, s in pairs),
                               alarm_episodes=len(own), cleared_episodes=sum(e['clear_frame'] is not None for e in own),
                               censored_episodes=sum(e['clear_frame'] is None for e in own)))
    switches = sum(1 + int(e['clear_frame'] is not None) for e in episodes)
    metric = dict(event_count=len(events), detected_events=len(matched),
                  event_recall=len(matched) / len(events) if events else 0.0,
                  normal_false_alarm_episodes=len(normal_episodes),
                  normal_feeding_decisions=normal_positive, normal_valid_decisions=normal_count,
                  normal_feeding_decision_ratio=normal_positive / normal_count if normal_count else None,
                  duplicate_event_alarms=sum(d['extra_alarm_episodes'] for d in details),
                  total_state_switches=switches,
                  mean_onset_delay_frames=sum(delays) / len(delays) if delays else None,
                  max_onset_delay_frames=max(delays) if delays else None,
                  matched_alarms_not_observed_clearing=uncleared,
                  mean_post_event_clear_delay_frames=sum(clear_delays) / len(clear_delays) if clear_delays else None,
                  false_alarms_per_hour=None, mean_onset_delay_seconds=None,
                  recovery_ground_truth_available=False)
    return metric, details, states, episodes, per_record


def selection_rank(metric, window):
    """Lexicographic, larger is better; no test input, no hidden accuracy priority."""
    def negative(value):
        return -float(value) if value is not None else -1e30
    return (metric['event_recall'], -metric['normal_false_alarm_episodes'],
            -metric['normal_feeding_decisions'], -metric['duplicate_event_alarms'],
            -metric['total_state_switches'], negative(metric['mean_onset_delay_frames']),
            -metric['matched_alarms_not_observed_clearing'],
            negative(metric['mean_post_event_clear_delay_frames']), window['f1'])
