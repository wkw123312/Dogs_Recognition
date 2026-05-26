import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np
from config import load_config

# 设置设备
device = torch.device("cpu")

config = load_config()


class DogBreedClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.breed_names = []
        self.transform = None
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        try:
            if not os.path.exists(self.model_path):
                return False

            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            self.breed_names = checkpoint['breed_names']

            # 创建模型架构
            model = models.mobilenet_v2(weights=None)
            num_ftrs = model.classifier[1].in_features

            # 尝试加载权重（修复键名）
            try:
                model.classifier[1] = nn.Linear(num_ftrs, len(self.breed_names))
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                # 如果失败，尝试复杂结构
                model.classifier[1] = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_ftrs, len(self.breed_names))
                )
                # 修复键名映射
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'classifier.1.1.' in key:
                        new_key = key.replace('classifier.1.1.', 'classifier.1.')
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict)

            model.to(device)
            model.eval()
            self.model = model

            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            return True

        except Exception as e:
            print(f"模型加载错误: {e}")
            return False

    def predict(self, image):
        """预测狗品种"""
        if self.model is None:
            return "模型未加载", {}, None

        try:
            # 转换图像格式
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                image = Image.open(image)

            image = image.convert('RGB')

            # 预处理
            input_tensor = self.transform(image).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)

                top_breed = self.breed_names[predicted.item()]
                top_confidence = confidence.item()

                # 获取前5个结果
                top5_probs, top5_indices = torch.topk(probabilities, 5)
                top5_results = []

                for i in range(5):
                    top5_results.append({
                        'breed': self.breed_names[top5_indices[i].item()],
                        'confidence': top5_probs[i].item()
                    })

                # 创建可视化结果
                result_text = f"🐕 最可能品种: {top_breed}\n"
                result_text += f"🎯 置信度: {top_confidence:.4f}\n\n"
                result_text += "🏆 前5个可能品种:\n"

                for i, result in enumerate(top5_results, 1):
                    result_text += f"{i}. {result['breed']}: {result['confidence']:.4f}\n"

                # 准备显示图像
                display_image = np.array(image)
                display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                # 调整大小以便显示
                height, width = display_image.shape[:2]
                max_size = 400
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_image = cv2.resize(display_image, (new_width, new_height))

                # 添加文字
                label = f"{top_breed.split('-')[-1]}: {top_confidence:.2f}"
                cv2.putText(display_image, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return result_text, top5_results, display_image

        except Exception as e:
            return f"预测错误: {str(e)}", {}, None


# 创建分类器实例
classifier = DogBreedClassifier(config["model_path"])


def predict_breed(image):
    """Gradio预测函数"""
    result_text, top5_results, display_image = classifier.predict(image)

    if display_image is not None:
        # 转换回RGB格式用于显示
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        return result_text, display_image
    else:
        return result_text, None


# 创建Gradio界面
with gr.Blocks(title="小狗识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐶 小狗识别系统")
    gr.Markdown("请上传小狗照片")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="请上传小狗图片", type="numpy")
            predict_btn = gr.Button("识别品种", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="识别结果", interactive=False)
            text_output = gr.Textbox(label="识别结果", lines=10)

    # 示例图片
    example_paths = config.get("example_images", [
        "./test_photo/img.png",
        "./test_photo/img_1.png"
    ])
    gr.Examples(
        examples=[[path] for path in example_paths],
        inputs=image_input,
        outputs=[text_output, image_output],
        fn=predict_breed,
        cache_examples=False
    )

    predict_btn.click(
        fn=predict_breed,
        inputs=image_input,
        outputs=[text_output, image_output]
    )

    # 添加使用说明
    with gr.Accordion("使用说明", open=False):
        gr.Markdown("""
        ## 使用方法：
        1. 点击"上传狗图片"或拖拽图片到上传区域
        2. 点击"识别品种"按钮
        3. 查看识别结果

        ## 支持功能：
        - 自动识别120种狗品种
        - 显示前5个最可能的品种
        - 实时显示识别置信度
        - 示例图片快速测试

        ## 注意事项：
        - 请上传清晰的狗正面照片
        - 系统目前准确率约30%，仍在优化中
        - 支持jpg、png等常见图片格式
        """)

if __name__ == "__main__":
    # 启动Gradio应用
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,  # 端口号
        share=False,  # 不创建公开链接
        debug=True  # 调试模式
    )
