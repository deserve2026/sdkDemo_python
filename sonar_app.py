"""Entry point: E:/hou_env/yazhou_py311/python.exe -B sonar_app.py [--source file.DB]."""
import sys
sys.dont_write_bytecode=True
import argparse
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication, QTabBar
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtCore import QTimer
from sonar_workbench.controller import Controller
from sonar_workbench.paths import output_path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-start',type=int,default=800)
    parser.add_argument('--smoke-end',type=int,default=879)
    args=parser.parse_args()
    app=QApplication(sys.argv[:1])
    # The offscreen Windows Qt backend has no system font inventory.
    if 'Microsoft YaHei' not in QFontDatabase.families():
        fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        for filename in ('msyh.ttc', 'msyhbd.ttc'):
            candidate = fonts / filename
            if candidate.is_file():
                QFontDatabase.addApplicationFont(str(candidate))
    app.setFont(QFont('Microsoft YaHei', 10))
    app.setApplicationName('SonarBehaviorWorkbench')
    controller=Controller(app,args.source,smoke=args.smoke)
    controller.window.show()
    if not args.smoke:
        area=app.primaryScreen().availableGeometry()
        controller.window.resize(min(1440,area.width()-40),min(940,area.height()-60))
    if args.smoke:
        import json
        def source_ready(_):
            controller.worker.command('seek',args.smoke_start)
            controller.worker.command('speed',40)
            controller.worker.command('play',True)
        controller.worker.source_ready.connect(source_ready)
        finished=[False]
        successful=[False]
        def check_layout(payload):
            dock=controller.docks[0]
            dock.setFloating(True)
            floating=dock.isFloating()
            dock.setFloating(False)
            controller.reset_layout()
            controller.docks[0].raise_()
            controller.window.resize(1200,820)
            def capture():
                responsive=(controller.widgets['imageLabel'].width()>=620
                            and controller.window.width()<=1200 and controller.window.height()<=820)
                tabs=[[bar.tabText(i) for i in range(bar.count())]
                      for bar in controller.window.findChildren(QTabBar) if bar.isVisible()]
                clean_tabs=tabs==[['摄食识别','画面与辅助工具']]
                font_ok='Microsoft YaHei' in QFontDatabase.families()
                screenshot=output_path('reports/sonar_workbench_preview.png')
                saved=controller.window.grab().save(str(screenshot))
                successful[0]=(floating and responsive and font_ok and saved and clean_tabs
                               and payload['result']['probability'] is not None)
                report=dict(passed=successful[0],frame=payload['index'],
                    model_state=payload['result']['state'],probability=payload['result']['probability'],
                    dock_float_supported=floating,responsive_resize=responsive,chinese_font_available=font_ok,
                    visible_tabs=tabs,window_size=[controller.window.width(),controller.window.height()],
                    screenshot=str(screenshot),ui_file=str(controller.ui_path))
                output_path('reports/qt_smoke_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(report,ensure_ascii=False),flush=True)
                QTimer.singleShot(100,controller.window.close)
            # Let Qt finish docking animations and deferred layout cleanup before inspection.
            QTimer.singleShot(300,capture)
        def frame_ready(payload):
            if not finished[0] and payload['index']>=args.smoke_end:
                finished[0]=True
                controller.worker.command('play',False)
                QTimer.singleShot(150,lambda:check_layout(payload))
        controller.worker.frame_ready.connect(frame_ready)
        def timeout():
            if not finished[0]:
                print('GUI smoke timed out',flush=True)
                controller.window.close()
                app.exit(2)
        QTimer.singleShot(90000,timeout)
    code=app.exec()
    controller.worker.shutdown()
    controller.worker.wait()
    return code if not args.smoke or successful[0] else 2

if __name__=='__main__':
    raise SystemExit(main())

