# 声呐行为分析工作台

本工作台基于现有 Python 3.11 / PySide6 工程，增加可编辑 Qt 界面和正常游动/摄食模型。
原 sonar_params.py 保留；新入口是 sonar_app.py。原有数量、体长功能可从工作台按钮进入。
新模型不识别应激，原有规则行为判断在工作台调用的旧工具入口中停用，避免混淆模型结果。

## 启动

使用原回放环境，不使用训练环境启动界面：

```powershell
& 'E:\hou_env\yazhou_py311\python.exe' -B 'E:\hou_py\sdkDemo_python\sonar_app.py'
```

也可以双击同目录的“启动声呐工作台.cmd”。
打开 `F:\experimental_data\9_12\投食\第5次投食\08.DB`；默认接续同名的08_1、08_2、08_3分段。
如只查看单个文件，在打开前取消“同名 DB 分段连续回放”。
从0开始完整回放用于复现；拖动进度或回到开头会重置模型。暂停/继续保留连续历史。
前32帧显示等待状态；首次判断在第31帧，后续每8帧判断一次。进入/退出均需连续2次。
回放速度是显示调度目标，解码/推理较慢时自动放慢；不丢掉模型输入帧来追赶速度。

## 编辑界面

“工具 → 在 Qt Designer 中编辑界面”打开 `sonar_workbench/ui/main_window.ui`。
可拖放按钮、图标、文字、布局和面板。保存后重启工作台生效，无需生成 ui_*.py。
已绑定控件的 objectName 请保持不变；增加全新功能按钮仍需要绑定Python处理逻辑。
样式位于 `sonar_workbench/theme.qss`。本次新主界面可在Designer编辑，原有旧工具仍由旧Python代码生成。

运行中拖动功能面板标题，可停靠、浮动或隐藏；“视图与布局”可恢复面板、恢复默认布局、锁定位置。
右侧默认使用标签页节省屏幕高度，可将任一面板拖出单独显示。
布局保存在结果目录 `db_replay/settings/layout.ini`，不写入原始数据或软件源码目录。

## 模型与图像约定

模型在 `F:\experimental_results\tcn\tcn_feed\onnx`：

- cnn_encoder.onnx：每帧图像 → 512维特征。
- tcn_classifier.onnx：32帧特征 → [normal, feeding] 概率。
- cnn_tcn_feeding.onnx：完整32帧图像输入模型，供后续C++对照。
- deployment.json / preprocess_config.yaml：权重指纹、输入形状、预处理与阈值。

实时回放使用前两个模型缓存每帧特征，不为相邻窗口反复运行同一帧CNN。
PNG不改变方向；SDK解码DB后上下翻转一次，匹配训练PNG。随后Pillow双线性缩放224×224，
灰度复制3通道、除255、按ImageNet均值/标准差归一化。显示翻转、检测框和文字不进入模型。
对齐检查：录制5前三个DB的全部3,666帧，与同编号训练PNG逐像素一致。
08_3.DB共223帧，未包含在原标注PNG中，属于无对应标注的补充回放段，不能自动计入原验证指标。
设备workCycle字段约0.1；未擅自将全部数据按40fps或10fps换算。原始采集时基仍需确认。

进入阈值0.8、退出0.6，均连续2次；参数来自验证集选择，当前界面不提供随手修改入口。
概率是模型得分，不是已经校准的现场正确率。5个验证事件检出并不等于已通过现场90%验收。

## 输出与环境

所有新输出都在 `F:\experimental_results\tcn\tcn_feed`：

- db_replay/indexes：按DB绝对路径、大小和修改时间区分的索引缓存。
- db_replay/sessions：每次回放逐帧预测、模型配置和输入文件信息。
- db_replay/settings：界面布局与原工具配置。
- reports：对齐检查、ONNX数值检查、Qt界面检查、回放核对报告和截图。

原始DB、PNG和标注CSV只读。索引不保存到DB旁边。回放环境已有onnxruntime，无需安装PyTorch。
导出使用 E:\hou_env\sonar_tcn；本次仅为导出补充onnx依赖，未更换训练权重。
当前为本地DB/PNG回放联调；没有接入实时声呐，也没有评估封存测试集。

## 检查命令

```powershell
& 'E:\hou_env\yazhou_py311\python.exe' -B -m unittest discover -s tests -v
& 'E:\hou_env\yazhou_py311\python.exe' -B tools/verify_onnx.py
```

跨DB拼接按实际索引长度累计，且要求设备frameSerialNum接续；不假设1200帧。
DB内部设备帧号不连续也会清空模型历史，重新收集窗口。
不支持的DB格式、解码失败、模型校验失败会暂停并显示错误，不用“正常游动”掩盖故障。

本次完整联调结果、人工复核帧段和环境变更见结果目录 `reports/qt_integration_report.md`。

旧数量/体长工具的两个检测模型位于 F:\experimental_results\mend26s\pt_onnx。直接运行旧入口使用项目内 sonar_params_config.json；从新工作台进入使用结果目录的 db_replay/settings/legacy_params.json，两处均已设置该模型目录。
