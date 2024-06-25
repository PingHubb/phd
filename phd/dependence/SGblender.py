import os
import sys
import json
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import subprocess

# 定义配置文件路径
config_file = 'conf.json'

# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 读取配置文件中的 Blender 路径
    with open(config_file, 'r') as f:
        data = json.load(f)
        blender_path =  data.get('blender_path', '')
    # blender_path = read_blender_path()

    if not os.path.exists(blender_path):
        QMessageBox.critical(None, "错误", "未找到保存的 Blender 可执行文件路径，请重新选择。")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        blender_path, _ = QFileDialog.getOpenFileName(None, "选择 Blender 可执行文件", "", "Blender Executable (*.exe);;All Files (*)", options=options)
        if blender_path:
            with open(config_file, 'w') as f:
                json.dump({'blender_path': blender_path}, f)
        else:
            QMessageBox.critical(None, "错误", "未选择 Blender 可执行文件路径，应用程序将退出。")
            sys.exit(1)
    
    # 运行你的主程序或者其他逻辑
    blender_args = [
    blender_path,
    "-b",  # 使用后台模式（无界面）
    "-P",  # 执行 Python 脚本
    os.getcwd() + r"\dependence\blender.py"
    ]
    subprocess.run(blender_args, check=True)
    sys.exit(1)

