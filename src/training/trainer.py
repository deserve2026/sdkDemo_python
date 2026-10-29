"""Train only on cached training features; select exclusively on validation replay."""
import json
import logging
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from src.data.features import load_caches
from src.data.feature_dataset import FeatureWindows, quality_valid
from src.models.tcn import build_tcn
from src.training.event_metrics import evaluate_events, window_metrics, selection_rank
from src.training.sampling import sample_epoch
from src.utils.experiment import source_hashes, verify_data
from src.utils.io import digest, read_csv, write_csv, write_json, write_text
from src.utils.paths import output_path


def predict(model, rows, caches, batch_size, device):
    indices = [i for i, r in enumerate(rows) if quality_valid(r)]
    loader = DataLoader(FeatureWindows([rows[i] for i in indices], caches), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    probabilities = [None] * len(rows)
    model.eval()
    offset = 0
    with torch.inference_mode():
        for x, _ in loader:
            values = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().tolist()
            for index, value in zip(indices[offset:offset + len(values)], values):
                probabilities[index] = value
            offset += len(values)
    if offset != len(indices):
        raise RuntimeError('Incomplete validation prediction')
    return probabilities


def save_checkpoint(path, payload):
    temporary = Path(path).with_suffix('.tmp.pth')
    torch.save(payload, temporary)
    temporary.replace(path)


def write_report(config, run_relative, best, history, train_rows, val_rows, run_id, elapsed):
    m, w = best['metrics'], best['window_metrics']
    counts = Counter(r['label'] for r in train_rows)
    text = f'''# 第一版固定 ResNet18 + 小型 TCN 实验

- 运行：{run_id}；训练完成，测试集尚未推理或用于选参。
- 使用 5 次摄食训练录制中的 20 个事件；验证为 1 次摄食录制、5 个事件及 2 次正常录制。
- 训练可用窗口：normal={counts['normal']}，feeding={counts['feeding']}。每轮全量摄食窗口、最多两倍正常窗口，正常样本按录制轮转抽样。
- 训练 {len(history)} 轮，耗时 {elapsed:.1f} 秒；选择第 {best['epoch']} 轮。
- 选择规则：事件检出率 → 正常误报次数 → 正常误报输出长度 → 重复报警 → 状态切换 → 起报延迟 → 报警退出表现 → 窗口 F1。
- 选定参数：{json.dumps(best['parameters'], ensure_ascii=False)}。

## 验证结果（用于选参，不能当作独立测试或现场验收）

| 指标 | 数值 |
|---|---:|
| 摄食事件检出率 | {m['detected_events']}/{m['event_count']} = {m['event_recall']:.1%} |
| 正常录制误报段数 | {m['normal_false_alarm_episodes']} |
| 正常录制显示摄食的决策点占比 | {m['normal_feeding_decision_ratio']:.2%} |
| 摄食事件额外报警段数 | {m['duplicate_event_alarms']} |
| 验证回放状态切换次数 | {m['total_state_switches']} |
| 平均起报延迟（帧） | {m['mean_onset_delay_frames']} |
| 最大起报延迟（帧） | {m['max_onset_delay_frames']} |
| 命中后未观测到退出的报警段数 | {m['matched_alarms_not_observed_clearing']} |
| 已退出报警的平均事件后退出延迟（帧） | {m['mean_post_event_clear_delay_frames']} |
| 窗口 feeding precision | {w['precision']:.4f} |
| 窗口 feeding recall | {w['recall']:.4f} |
| 窗口 feeding F1 | {w['f1']:.4f} |
| 窗口 accuracy（辅助） | {w['accuracy']:.4f} |

混淆矩阵：{w['confusion_matrix']}；行是真实 normal/feeding，列是预测 normal/feeding。
窗口指标对已知标签使用选定进入阈值；事件指标使用有进入/退出滞回及连续两次确认的界面状态。

## 逐事件

| recording_key | event_id | 检出 | 判断帧 | 延迟帧 | 重复报警 |
|---|---|---|---|---|---|
'''
    for d in best['events']:
        text += f"| {d['recording_key']} | {d['event_id']} | {d['detected']} | {d['decision_frame']} | {d['onset_delay_frames']} | {d['extra_alarm_episodes']} |\n"
    text += '\n## 逐录制回放\n\n| recording_key | 有效决策点 | 显示摄食点 | 报警段 | 已退出段 | 未观测退出段 |\n|---|---|---|---|---|---|\n'
    for r in best['recordings']:
        text += f"| {r['recording_key']} | {r['valid_decisions']} | {r['feeding_decisions']} | {r['alarm_episodes']} | {r['cleared_episodes']} | {r['censored_episodes']} |\n"
    text += '''
## 口径和限制

- 一个持续报警最多匹配一个事件，一个事件最多获得一次成功；判定使用窗口末帧，不回填到起始帧。
- 所有画面有效且连续的验证窗口都会推理，包括未知标签 ignore；未知摄食来源区间不计作正常误报。
- 坏图或跨断点窗口不推理，并重置状态；录制之间不继承状态。无效决策点的 ui_state=-1，代表数据无效。
- 摄食结束后区间尚未确认全部正常。“退出延迟”仅描述显示何时退出，不能等同于有真值的恢复正常准确率；事件内提前退出也可能得到0，须结合回放判断。
- 正常显示摄食占比是按步长8采样的决策点比例，不是假定精确物理时长。
- FPS 未核实：秒级延迟、每小时误报均为 null，不以40帧/秒代算。
- 验证集只有5个事件且被用于选参；即使5/5也不能证明现场达到90%。
- 未评估测试集、未导出ONNX。正负来源分离可能引入背景/增益等录制差异，仍需后续按录制复核。
- 各轮和每个阈值的结果完整保存在 history.csv、threshold_search.csv；已选配置保存在 selected_model.json。
'''
    write_text(config, run_relative / 'train_report.md', text)
    write_text(config, 'reports/train_report.md', text)


def train(config, model_config, settings, project_root):
    if not settings.get('enabled') or not settings.get('data_review_accepted') or settings.get('test_for_tuning'):
        raise ValueError('Training authorization/configuration is invalid')
    if not settings.get('balance_train_only') or settings['epochs'] < 1:
        raise ValueError('Invalid training settings')
    if model_config['window_size'] != config['window_size']:
        raise ValueError('Window sizes disagree')
    verify_data(config)
    caches, cache_manifest = load_caches(config, model_config)
    train_rows = [r for r in read_csv(Path(config['windows_dir']) / 'windows_train.csv') if r['label'] in ('normal', 'feeding')]
    val_rows = read_csv(Path(config['windows_dir']) / 'windows_val.csv')
    val_rows.sort(key=lambda r: (r['recording_key'], int(r['start_global_frame'])))
    if any(r['split'] != 'train' for r in train_rows) or any(r['split'] != 'val' for r in val_rows):
        raise ValueError('Split contamination')
    val_keys = {r['recording_key'] for r in val_rows}
    if val_keys & {r['recording_key'] for r in train_rows}:
        raise ValueError('Recording leakage')
    events = [e for e in read_csv(Path(config['metadata_dir']) / 'events_clean.csv') if e['recording_key'] in val_keys]
    if not events or not any(r['behavior_source'] == 'normal' for r in val_rows):
        raise ValueError('Validation requires feeding events and confirmed normal recordings')
    seed = settings['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(settings.get('cpu_threads', 4))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings['device'])
    model = build_tcn(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    loss_function = nn.CrossEntropyLoss()
    dataset = FeatureWindows(train_rows, caches)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    relative = Path('runs') / run_id
    run_dir = output_path(config, relative / 'protocol.json').parent
    protocol = dict(run_id=run_id, model_config=model_config, train_config=settings,
                    dataset_config_sha256=config['_config_sha256'], source_sha256=source_hashes(project_root),
                    cache_manifest=str(cache_manifest), cache_manifest_sha256=digest(cache_manifest),
                    train_csv_sha256=digest(Path(config['windows_dir']) / 'windows_train.csv'),
                    val_csv_sha256=digest(Path(config['windows_dir']) / 'windows_val.csv'),
                    parameter_count=sum(p.numel() for p in model.parameters()), test_used=False,
                    ranking=['event_recall descending', 'normal_false_alarm_episodes ascending',
                             'normal_feeding_decisions ascending', 'duplicate_event_alarms ascending',
                             'total_state_switches ascending', 'mean_onset_delay_frames ascending',
                             'matched_alarms_not_observed_clearing ascending',
                             'mean_post_event_clear_delay_frames ascending', 'window_f1 descending'],
                    on_exact_tie='earliest epoch then first threshold in configured order',
                    unavailable_delay_rank='worst', fps=config['fps'])
    write_json(config, relative / 'protocol.json', protocol)
    for subdir in ('src', 'scripts', 'configs'):
        shutil.copytree(Path(project_root) / subdir, run_dir / 'source_snapshot' / subdir,
                        ignore=shutil.ignore_patterns('__pycache__'))
    status_path = Path(config['metadata_dir']) / 'pipeline_status.json'
    pipeline = json.loads(status_path.read_text(encoding='utf-8'))
    pipeline.update(training_started=True, training_run_id=run_id)
    write_json(config, 'metadata/pipeline_status.json', pipeline)
    write_json(config, 'metadata/training_status.json', dict(status='running', run_id=run_id, test_evaluated=False))
    history, searches, best, best_rank = [], [], None, None
    seen_normal = set()
    started = time.perf_counter()
    try:
        for epoch in range(1, settings['epochs'] + 1):
            indices = sample_epoch(train_rows, settings['normal_to_feeding_max_ratio'], seed + epoch)
            seen_normal.update(train_rows[i]['window_id'] for i in indices if train_rows[i]['label'] == 'normal')
            loader = DataLoader(Subset(dataset, indices), batch_size=settings['batch_size'], shuffle=False,
                                num_workers=0)
            model.train()
            loss_sum, samples = 0.0, 0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item() * len(y)
                samples += len(y)
            probabilities = predict(model, val_rows, caches, settings['batch_size'], device)
            epoch_best = None
            for entry in settings['entry_thresholds']:
                parameters = dict(entry=float(entry), exit_threshold=round(float(entry) - settings['exit_threshold_gap'], 6),
                                  entry_count=settings['entry_consecutive'], exit_count=settings['exit_consecutive'])
                metric, details, states, episodes, recordings = evaluate_events(val_rows, probabilities, events, parameters, config['stride'])
                window = window_metrics(val_rows, probabilities, entry)
                rank = selection_rank(metric, window)
                searches.append(dict(epoch=epoch, **parameters, **metric, window_f1=window['f1']))
                if epoch_best is None or rank > epoch_best[0]:
                    epoch_best = (rank, metric, entry)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best = dict(epoch=epoch, parameters=parameters, metrics=metric, window_metrics=window,
                                events=details, recordings=recordings, rank=list(rank))
                    save_checkpoint(run_dir / 'best_model.pth', dict(model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                    model_config=model_config, selection=best, protocol=protocol))
                    predictions = []
                    for row, probability, state in zip(val_rows, probabilities, states):
                        predictions.append(dict(**row, true_label=row['label'], feeding_prob=probability,
                                                pred_label=('feeding' if probability >= entry else 'normal') if probability is not None else 'invalid',
                                                ui_state=state, decision_global_frame=int(row['end_global_frame'])))
                    write_csv(config, relative / 'val_predictions.csv', predictions, list(predictions[0]))
                    write_json(config, relative / 'alarm_episodes.json', episodes)
            entry_counts = Counter(train_rows[i]['label'] for i in indices)
            history.append(dict(epoch=epoch, train_loss=loss_sum / samples, train_normal=entry_counts['normal'],
                                train_feeding=entry_counts['feeding'], unique_normal_seen=len(seen_normal),
                                epoch_best_entry=epoch_best[2], **epoch_best[1]))
            write_csv(config, relative / 'history.csv', history, list(history[0]))
            write_csv(config, relative / 'threshold_search.csv', searches, list(searches[0]))
            logging.info('Epoch %d/%d loss=%.4f; val events=%d/%d normal FP=%d; selected epoch=%d threshold=%.2f',
                         epoch, settings['epochs'], loss_sum / samples, epoch_best[1]['detected_events'], len(events),
                         epoch_best[1]['normal_false_alarm_episodes'], best['epoch'], best['parameters']['entry'])
        best.update(run_id=run_id, test_evaluated=False, selection_frozen=True,
                    checkpoint=str(run_dir / 'best_model.pth'), checkpoint_sha256=digest(run_dir / 'best_model.pth'),
                    protocol_sha256=digest(run_dir / 'protocol.json'))
        write_json(config, relative / 'selected_model.json', best)
        write_json(config, 'checkpoints/selected_model.json', best)
        write_csv(config, relative / 'val_event_results.csv', best['events'], list(best['events'][0]))
        write_csv(config, relative / 'val_recording_results.csv', best['recordings'], list(best['recordings'][0]))
        destination = output_path(config, 'checkpoints/best_model.pth')
        if destination.exists():
            shutil.copy2(destination, run_dir / 'previous_best_model.pth')
        temporary = destination.with_suffix('.copying')
        shutil.copy2(run_dir / 'best_model.pth', temporary)
        temporary.replace(destination)
        shutil.copy2(run_dir / 'val_predictions.csv', output_path(config, 'predictions/val_predictions.csv'))
        write_report(config, relative, best, history, train_rows, val_rows, run_id, time.perf_counter() - started)
        write_json(config, 'metadata/training_status.json', dict(status='completed', run_id=run_id,
                   selected_model=str(run_dir / 'selected_model.json'), test_evaluated=False))
        logging.info('Training complete. Selected val events=%d/%d, normal false alarms=%d. Test untouched.',
                     best['metrics']['detected_events'], len(events), best['metrics']['normal_false_alarm_episodes'])
        return best
    except Exception as error:
        write_json(config, 'metadata/training_status.json', dict(status='failed', run_id=run_id, error=str(error), test_evaluated=False))
        raise

