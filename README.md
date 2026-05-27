# 小狗识别系统

## Description

本项目是一个基于 PyTorch 和 Gradio 的狗品种识别系统，使用 Stanford Dogs Dataset 训练模型。它支持从配置文件读取模型路径和测试图片路径，避免在代码中硬编码个人服务器绝对路径。

## Installation

1. 克隆仓库到本地目录：
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. 创建并激活虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. 复制配置模板：
   ```bash
   cp config.yaml.example config.yaml
   ```
2. 编辑 `config.yaml`，将路径改为你的本地项目路径，例如：
   - `model_path: ./runs/dog_breed_classifier_cpu.pth`
   - `test_images_dir: ./test_photo`
   - `dataset_root: ./dog_breeds_dataset`
3. 启动 Gradio 应用：
   ```bash
   python Gradio_test.py
   ```
4. 你也可以直接在 `test_model.py` 中使用 `config.yaml`指定的模型路径进行命令行测试。

## Configuration

`config.yaml.example` 包含以下字段：

- `model_path`: 模型权重文件路径
- `device`: 设备类型，如 `cpu` 或 `cuda`
- `test_images_dir`: 默认测试图片目录
- `example_images`: Gradio 界面示例图片列表
- `dataset_root`: 原始 Stanford Dogs 数据集根目录
- `output_dir`: 模型输出目录

## Disclaimer

本项目代码已做脱敏处理，已移除个人服务器硬编码路径。使用前请自行创建 `config.yaml` 并配置本地路径，确保所有路径指向合法的本地目录。请勿将个人配置文件、模型权重或私有数据提交到公共仓库。

## License

请参阅 `LICENSE` 文件。
