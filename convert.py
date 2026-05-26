import os
import shutil
from sklearn.model_selection import train_test_split


def organize_stanford_dataset(original_path, output_path):
    """
    整理斯坦福数据集为PyTorch需要的格式

    原始结构:
    Images/
        breed1/
            image1.jpg
            image2.jpg
        breed2/
            image1.jpg
            ...

    目标结构:
    dog_breeds/
        train/
            breed1/
                image1.jpg
                image2.jpg
            breed2/
                ...
        val/
            breed1/
                image1.jpg
                ...
    """

    # 创建输出目录
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有品种文件夹
    breed_folders = [f for f in os.listdir(original_path)
                     if os.path.isdir(os.path.join(original_path, f))]

    print(f"找到 {len(breed_folders)} 个品种")

    for breed_folder in breed_folders:
        breed_path = os.path.join(original_path, breed_folder)

        # 获取品种名称（从文件夹名中提取，如：n02085620-Chihuahua -> Chihuahua）
        breed_name = breed_folder.split('-')[1] if '-' in breed_folder else breed_folder

        # 获取该品种的所有图片
        image_files = [f for f in os.listdir(breed_path)
                       if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            continue

        # 划分训练集和验证集
        train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

        # 创建品种文件夹
        breed_train_dir = os.path.join(train_dir, breed_name)
        breed_val_dir = os.path.join(val_dir, breed_name)
        os.makedirs(breed_train_dir, exist_ok=True)
        os.makedirs(breed_val_dir, exist_ok=True)

        # 复制训练集图片
        for img_file in train_files:
            src = os.path.join(breed_path, img_file)
            dst = os.path.join(breed_train_dir, img_file)
            shutil.copy2(src, dst)

        # 复制验证集图片
        for img_file in val_files:
            src = os.path.join(breed_path, img_file)
            dst = os.path.join(breed_val_dir, img_file)
            shutil.copy2(src, dst)

        print(f"处理完成: {breed_name} (训练: {len(train_files)}, 验证: {len(val_files)})")

    print("数据集整理完成！")


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 Stanford Dogs Dataset 整理为训练/验证目录结构")
    parser.add_argument("--source", default="./Images", help="原始 Stanford Dogs 数据集 Images 目录")
    parser.add_argument("--output", default="./dog_breeds_dataset", help="整理后的输出目录")
    args = parser.parse_args()

    organize_stanford_dataset(args.source, args.output)
