import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
from config import load_config

# 设置设备
device = torch.device("cpu")

config = load_config()


def load_trained_model(model_path):
    """加载训练好的模型"""
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        breed_names = checkpoint['breed_names']
        num_breeds = checkpoint['num_breeds']

        print(f"✅ 模型加载成功！支持 {num_breeds} 个狗品种")

        # 创建模型架构
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features

        # 尝试两种结构
        try:
            # 先尝试简单结构
            model.classifier[1] = nn.Linear(num_ftrs, num_breeds)
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # 如果失败，尝试复杂结构并修复键名
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_ftrs, num_breeds)
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

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return model, breed_names, transform

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None


def predict_dog_breed(image_path, model, breed_names, transform):
    """预测狗品种"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            return None

        # 读取和预处理图片
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

            # 获取前5个结果
            top5_probs, top5_indices = torch.topk(probabilities, 5)

            results = {
                'top_breed': breed_names[predicted.item()],
                'top_confidence': confidence.item(),
                'top5': []
            }

            for i in range(5):
                results['top5'].append({
                    'breed': breed_names[top5_indices[i].item()],
                    'confidence': top5_probs[i].item()
                })

            return results

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return None


def display_results(results, image_path):
    """显示结果"""
    if results is None:
        return

    print("\n" + "=" * 50)
    print("🐕 狗品种识别结果")
    print("=" * 50)
    print(f"图片: {os.path.basename(image_path)}")
    print(f"最可能品种: {results['top_breed']}")
    print(f"置信度: {results['top_confidence']:.4f}")

    print("\n🏆 前5个可能品种:")
    for i, result in enumerate(results['top5'], 1):
        print(f"{i}. {result['breed']}: {result['confidence']:.4f}")

    # 显示图片
    try:
        image = cv2.imread(image_path)
        if image is not None:
            # 调整图片大小以便显示
            height, width = image.shape[:2]
            max_size = 600
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            # 添加文字
            label = f"{results['top_breed']}: {results['top_confidence']:.2f}"
            cv2.putText(image, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Dog Breed Recognition", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except:
        print("⚠️  无法显示图片，但预测已完成")


def main():
    # 加载模型
    model_path = config.get("model_path")
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在，请确保 config.yaml 中的 model_path 指向正确文件：{model_path}")
        return

    model, breed_names, transform = load_trained_model(model_path)
    if model is None:
        return

    # 获取要测试的图片路径
    print("\n📁 请提供要测试的狗图片路径")
    print("例如: C:/Users/YourName/Pictures/dog.jpg")
    print("或者直接将图片拖拽到终端窗口中")

    while True:
        image_path = input("\n请输入图片路径: ").strip().strip('"')

        if image_path.lower() == 'quit':
            break

        if not image_path:
            continue

        # 预测
        results = predict_dog_breed(image_path, model, breed_names, transform)

        if results:
            display_results(results, image_path)

        print("\n输入 'quit' 退出，或继续测试其他图片")


if __name__ == "__main__":
    main()
