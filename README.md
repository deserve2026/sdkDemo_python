# 声呐鱼群正常游动 / 摄食识别

已实现数据管线和经用户确认的第一版实验：冻结 ResNet18 提取逐帧特征，再训练小型因果 TCN。
extract_features.py 默认只提取 train/val；train.py 只用训练集学习、验证集选模型和阈值。
evaluate.py 仍为占位入口，测试集保持封存。export_onnx.py 已实现；DB回放软件使用单独的yazhou_py311环境和ONNX Runtime。

## 路径

- Python环境：`E:\hou_env\sonar_tcn`
- 代码唯一目录：`E:\hou_py\sonar_tcn`
- 摄食源数据：`F:\experimental_data\数据标注\gts（高体鰤）\摄食（feed）`
- 正常源数据：`F:\experimental_data\数据标注\gts（高体鰤）\正常游动（normal）`
- 摄食标注：摄食源目录中的`feeding_events.bak.csv`（明确指定，不选择其他备份）。
- 全部程序输出：`F:\experimental_results\tcn\tcn_feed`

源PNG、DB和人工CSV只读，不移动、删除、覆盖或改名。代码使用pathlib.Path；中文图片路径可使用
`src.utils.image.imread_unicode`（numpy.fromfile + cv2.imdecode），不依赖cv2.imread。
程序日志、CSV、报告、未来的权重和特征缓存都必须在结果根目录。入口禁用代码目录中的字节码缓存。

## 结构

```text
configs/       dataset.yaml / model.yaml / train.yaml
scripts/       audit_dataset.py / build_timeline.py / clean_events.py
               make_split.py / make_windows.py / run_pipeline.py
               extract_features.py / train.py / evaluate.py / export_onnx.py（后两者占位）
src/data/      audit.py / timeline.py / events.py / splitting.py / windowing.py / dataset.py
               features.py / feature_dataset.py
src/utils/     io.py / image.py / paths.py / logging.py
src/models/    cnn_backbone.py / tcn.py / cnn_tcn.py
src/training/  trainer.py / event_metrics.py / sampling.py / metrics.py / losses.py
tests/         有意义的映射、缺帧、标签和划分检查
```

## 环境

已存在的环境不需要重建。首次新建时：

```bat
conda create --prefix E:\hou_env\sonar_tcn python=3.11 -y
conda activate E:\hou_env\sonar_tcn
python -m pip install --upgrade pip
cd /d E:\hou_py\sonar_tcn
python -m pip install -r requirements.txt
```

PyTorch按本机显卡/驱动到 https://pytorch.org/get-started/previous-versions/ 选择官方匹配命令。
本机已核对的组合为Python 3.11、torch 2.10.0、torchvision 0.25.0、cu128：

```bat
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA版本不写入requirements.txt。不需要torchaudio；当前也不需要ONNX依赖。
PyTorch成功安装不等于已下载预训练模型。后续显式调用cnn_backbone.make_feature_extractor时，
torchvision自动下载ResNet18预训练权重；torch.hub缓存已重定向到结果目录的checkpoints/pretrained，
不要求手动下载，也不在审计阶段下载。

## 运行数据管线

Anaconda Prompt中执行。以下命令按依赖顺序排列：

```bat
conda activate E:\hou_env\sonar_tcn
cd /d E:\hou_py\sonar_tcn
python -B -m unittest discover -s tests -v
python -B scripts/audit_dataset.py
python -B scripts/build_timeline.py
python -B scripts/clean_events.py
python -B scripts/make_split.py
python -B scripts/make_windows.py
```

也可以只运行一次：

```bat
python -B scripts/run_pipeline.py
```

入口默认从自身代码位置寻找configs/dataset.yaml；支持`--config`显式指定配置。
报告和CSV为可重建输出，重新运行会更新对应结果文件。CSV使用UTF-8-SIG，路径列表JSON和Markdown使用UTF-8。
上游重新运行将使下游通过状态失效；配置、CSV与源文件清单发生变化时不得复用旧窗口。
任何阶段失败都返回非0状态并留下报告。pipeline_status.json的data_ready_for_training只是自动数据检查结果，
不意味着已经做过模型训练或达到90%事件检出率。

## 已确认的数据规则

1. gts_feed_1是录制1第0段，gts_feed_1_2是录制1第2段。按数字段号排序，绝不按字符串顺序拼接。
2. 每段按实际帧号排序；global_frame是从0开始的实际文件顺序索引，不是物理时间。
3. 段间接续依据用户确认；不假设每段1200帧。缺帧、坏图、缺段、新段非0起始会产生连续性断点。
4. 事件和窗口端点都是闭区间，32帧窗口end=start+31。
5. 先按(behavior, recording_id)分组并固定划分，再生成窗口。不同录制绝不拼接。
6. 正常录制整体标注正常，无需正常事件起止帧；normal_events_csv=null，已有正常CSV不是必需输入。
7. 正常源中的有效完整窗口标normal；摄食源中已标注摄食帧占比>=80%标feeding，其余全部ignore。
8. 质量无效或跨连续性断点的窗口优先ignore；摄食源未标注部分绝不自动成为normal。
9. 无额外1秒边界忽略带；fps=null，40只作未核实备注，不据此计算秒级延迟或每小时误报。
10. 仅有species内容、六个事件核心字段全空的CSV行视为占位行。部分字段缺失的事件是错误，不能当空行删掉。
11. 预期34事件用于核对，不能强行凑数；端点无效、跨录制或严重事件重叠都会阻止下游。
12. 帧级重叠按事件并集计算；逐事件检查是否有正窗口，不能只看总窗口数。

摄食固定划分：train=2,3,4,6,7；val=5；test=1,8。
正常固定划分：train=2,3,4,6,7,9,10,11；val=5,12；test=1,8,13。

## 输出与重建

- metadata/recordings.csv：分段信息。
- metadata/frame_manifest.csv：逐帧SHA-256、大小、修改时间、解码状态和尺寸。
- metadata/timeline_index.csv / timeline_summary.csv：全帧索引和录制摘要。
- metadata/events_clean.csv：已验证的事件及原文件/行号溯源。
- metadata/splits.csv：固定录制划分。
- metadata/path_lists/*.json：每次录制共用一个图片路径数组。
- metadata/pipeline_status.json：各阶段状态、输入/输出指纹及自动验收状态。
- quality_reports/quality_report.csv / audit_summary.md：图像质量审计。
- quality_reports/events_cleaning_report.md / events_rejected.csv / events_removed_rows.csv / event_overlaps.csv。
- quality_reports/cross_split_duplicate_images.csv：字节级相同图像跨集合检查。
- windows/windows_train.csv / windows_val.csv / windows_test.csv。
- reports/split_report.md / window_report.md / window_recording_counts.csv / event_window_coverage.csv。
- logs/：数据阶段日志。

窗口CSV中的image_path_list_file指向所在录制共享的JSON数组，
通过`paths[start_global_frame:end_global_frame+1]`重建。也可调用src.data.dataset.reconstruct_paths。
不为每个重叠窗口复制PNG或创建一个小文件。所有ignore窗口都保留，训练时只读取normal和feeding。

## 为什么不能随机划分帧或窗口

相邻帧高度相似，32帧/步长8的窗口还共享大量图片。随机划分会让同次录制的图像同时出现在训练和测试，
导致看似高分但不能代表新录制效果。必须保证同次真实采集、同一recording的所有分段在同一个集合。
这版只按用户确认的一一对应来源编号分组；若以后发现不同编号来自同一次原始采集，必须重新定义组。
跨集合字节相同图像仅报告并阻断，不擅自删除。任何从数据学习的归一化统计只能来自训练集。

## 第一版特征缓存和训练

在数据报告确认后运行（本次用户已经确认）：

```bat
python -B scripts/extract_features.py --splits train val
python -B scripts/train.py
```

ResNet18 的 IMAGENET1K_V1 权重首次自动下载到结果目录；模型 eval、参数冻结，逐帧提取512维 float32 特征。
每张有效图片先核对审计SHA-256，再由Pillow灰度化、双线性缩放224×224、复制3通道、除255并使用固定ImageNet均值/标准差。
采用整幅缩放，无裁剪、无训练增强；后续C++必须保持处理一致。
每次录制一个NPY缓存，坏图保留NaN占位，不压缩索引。缓存按图像清单、配置、代码和依赖版本生成指纹，
每个已完成录制可校验后复用；中断后当前未完成录制重新提取，其他录制不重复提取。

TCN使用每帧LayerNorm、512→64投影、4个双卷积残差块（通道64、核3、膨胀1/2/4/8），取窗口末位置输出二分类。
每轮使用全部244个摄食窗口和488个正常窗口；正常样本跨8次训练录制均衡抽取，每轮改变且单轮不重复。
交叉熵不加类别权重，AdamW，30轮。不会对验证集抽样平衡。

进入阈值候选0.4/0.5/0.6/0.7/0.8，退出阈值=进入阈值−0.2；进入/退出都需连续2次。
验证集完整时序回放，包括画面有效的ignore窗口；质量异常窗口重置状态。
按事件检出率最大、正常误报段数最少、正常显示摄食点最少、重复报警最少、切换最少、起报延迟最短、
未观测退出数最少、退出延迟最短、窗口F1最大，依次选择；完全相同保留较早轮次/配置中较早阈值。
每个持续报警最多匹配一个真实事件；事件内第一次实际输出摄食的窗口末帧为判定时刻，不提前回填。
这里的事件后退出延迟是显示行为描述，摄食结束后没有确认正常真值，不能据此声称恢复正常准确率。

结果输出：

- features/<指纹>/manifest.json、<behavior>_<recording>.npy：特征和完整性信息。
- runs/<UTC时间>/：不可混淆的独立运行，包含protocol.json、源码快照、history.csv、threshold_search.csv、
  best_model.pth、selected_model.json、val_predictions.csv、val_event_results.csv、val_recording_results.csv、alarm_episodes.json和train_report.md。
- checkpoints/best_model.pth：当前选定的TCN权重和模型配置（冻结CNN权重另存于pretrained）。
- checkpoints/selected_model.json：已固定的轮次、阈值和验证结果。
- predictions/val_predictions.csv、reports/train_report.md：方便查阅的当前结果。
- metadata/training_status.json：是否完成、运行编号、test_evaluated=false。

若再次显式运行训练，会创建新run并更新当前结果；先前run保留。不要在看过测试集后反复运行调参。
本轮无需新增依赖，使用已安装torch/torchvision及数据阶段依赖；不需要psutil、ONNX或TensorBoard。

## 后续独立评估、导出

测试入口仍为占位，当前会明确退出：

```bat
python -B scripts/evaluate.py
```

首期验收关注真实摄食事件检出率>=90%以及显示不反复跳动；次期>=95%。当前不把严格起止延迟设为通过门槛，
但仍记录漏检、误报、延迟、恢复正常及重复报警；不能把始终显示摄食当作成功。
不允许使用test反复调参。当前固定test包含9个摄食事件，现场10次验收需另外安排。

训练集可按normal:feeding不超过2:1抽样；验证和测试不平衡采样。
未知摄食源区间不能计作已确认正常的误报分母；在线回放仍连续预测，不能删掉ignore时间点。

未来C++部署使用ONNX Runtime，保存并一致实现灰度转RGB、PIL双线性缩放的等效处理、归一化、
32帧缓存、预测步长和状态切换规则。具体数值一致性必须用同一组输入比较Python与C++输出。
已进入ONNX和DB回放联调阶段。训练环境只补充requirements-export.txt中的导出依赖，保持原PyTorch版本。
运行 `python -B scripts/export_onnx.py` 输出完整模型和分离的CNN、TCN模型，以及预处理配置。
导出初始状态verified=false；使用回放工程tools/verify_onnx.py完成真实验证窗口数值对照后才启用。
完整模型输入[1,32,3,224,224]，输出[normal,feeding]概率；回放使用每帧CNN+32帧特征缓存避免重复计算。
结果在F:\experimental_results\tcn\tcn_feed\onnx，未改动训练参数或使用测试集调参。

## 已训练模型的复核

`python -B scripts/verify_experiment.py` 在CPU重新加载已选模型，验证全部验证预测、界面状态和事件指标；核对缓存完整性，并生成 reports/val_replay.png、training_curve.png 和 experiment_verification.json。该命令不重新训练、不选择新参数、不读取测试预测。
