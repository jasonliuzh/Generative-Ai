import os
from PIL import Image
import cv2
import numpy as np
from collections import Counter

def crop_color_square(input_folder):
    """裁剪图像中的目标颜色区域"""
    TARGET_COLOR = (49, 49, 49)  # #313131的RGB值

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not os.path.isfile(input_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            img = Image.open(input_path).convert('RGB')
            width, height = img.size  # 获取图像的宽度和高度

            # 初始化四个基准点
            first_top = None  # (x, y) 从上到下第一个像素（最小y，相同y取最小x）
            last_bottom = None  # (x, y) 从上到下最后一个像素（最大y，相同y取最大x）
            first_left = None  # (x, y) 从左到右第一个像素（最小x，相同x取最小y）
            last_right = None  # (x, y) 从左到右最后一个像素（最大x，相同x取最大y）

            pixels = img.load()
            for y in range(height):
                for x in range(width):
                    if pixels[x, y] == TARGET_COLOR:
                        # 更新从上到下第一个像素
                        if first_top is None or y < first_top[1] or (y == first_top[1] and x < first_top[0]):
                            first_top = (x, y)

                        # 更新从上到下最后一个像素
                        if last_bottom is None or y > last_bottom[1] or (y == last_bottom[1] and x > last_bottom[0]):
                            last_bottom = (x, y)

                        # 更新从左到右第一个像素
                        if first_left is None or x < first_left[0] or (x == first_left[0] and y < first_left[1]):
                            first_left = (x, y)

                        # 更新从左到右最后一个像素
                        if last_right is None or x > last_right[0] or (x == last_right[0] and y > last_right[1]):
                            last_right = (x, y)

            # 检查是否找到有效像素
            if not all([first_top, last_bottom, first_left, last_right]):
                print(f"未找到完整基准点: {filename}")
                continue

            # 计算正方形边界
            x1 = first_left[0] - 4
            y1 = first_top[1] - 4
            x2 = last_right[0] + 4
            y2 = last_bottom[1] + 4

            # 确保裁剪区域在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            # 执行裁剪
            cropped = img.crop((x1, y1, x2, y2))
            print(f"成功裁剪: {filename} [尺寸: {cropped.size}]")

            # 返回裁剪后的图像
            yield filename, cropped

        except Exception as e:
            print(f"处理失败 {filename}: {str(e)}")

def get_most_common_color(block):
    """获取 9x9 区域中最常见的颜色"""
    # 将区域转换为二维数组
    pixels = block.reshape(-1, 3)
    # 统计每种颜色的出现次数
    color_counts = Counter(map(tuple, pixels))
    # 返回出现次数最多的颜色
    return np.array(color_counts.most_common(1)[0][0], dtype=np.uint8)

def reorganize_image(image, block_size=9, stride=3):
    """重组图片"""
    height, width = image.shape[:2]
    # 计算新图像的尺寸
    new_height = (height - block_size) // stride + 1
    new_width = (width - block_size) // stride + 1
    # 初始化新图像
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 遍历每个 block_size x block_size 区域
    for i in range(0, height - block_size + 1, stride):
        for j in range(0, width - block_size + 1, stride):
            # 获取当前 block_size x block_size 区域
            block = image[i:i + block_size, j:j + block_size]
            # 获取区域中最常见的颜色
            most_common_color = get_most_common_color(block)
            # 将颜色赋值给新图像的对应位置
            new_i, new_j = i // stride, j // stride
            new_image[new_i, new_j] = most_common_color

    return new_image

def pixelate_image(image, canvas_size=(32, 32)):
    """
    将图片智能缩放后居中放置在指定画布中，保持清晰的像素化效果
    :param image: PIL.Image 对象
    :param canvas_size: 画布尺寸（默认32x32）
    :return: 像素化后的 PIL.Image 对象
    """
    try:
        # 原始图片尺寸
        orig_width, orig_height = image.size

        # 第一步：将图片放在的画布上
        # 创建白色背景画布
        temp_canvas_size = (139, 139)
        temp_canvas = Image.new("RGB", temp_canvas_size, (255, 255, 255))

        # 计算153x153画布上的居中位置
        temp_position = (
            (temp_canvas_size[0] - orig_width) // 2,
            (temp_canvas_size[1] - orig_height) // 2
        )

        # 将原始图片粘贴到153x153画布中央
        temp_canvas.paste(image, temp_position)

        # 第二步：对153x153画布上的图片进行缩放和居中
        # 计算缩放比例
        width_ratio = canvas_size[0] / temp_canvas_size[0]
        height_ratio = canvas_size[1] / temp_canvas_size[1]
        scale_ratio = min(width_ratio, height_ratio)

        # 计算新尺寸（确保新尺寸是整数）
        new_size = (
            max(1, int(temp_canvas_size[0] * scale_ratio)),  # 确保至少为1像素
            max(1, int(temp_canvas_size[1] * scale_ratio))
        )

        # 使用最近邻插值保持像素化效果
        resized = temp_canvas.resize(new_size, Image.Resampling.NEAREST)

        # 创建最终白色背景画布
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))

        # 计算粘贴位置（确保居中且无偏移）
        position = (
            (canvas_size[0] - new_size[0]) // 2,
            (canvas_size[1] - new_size[1]) // 2
        )

        # 将缩放后的图片粘贴到画布中央
        canvas.paste(resized, position)
        return canvas
    except Exception as e:
        print(f"像素化处理失败: {str(e)}")
        return None

def process_folder(input_folder, output_folder):
    """处理文件夹中的所有图像"""
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历裁剪后的图像
    for filename, img in crop_color_square(input_folder):
        try:
            # 将 PIL 图像转换为 NumPy 数组（OpenCV 格式）
            image_np = np.array(img)
            # 将 RGB 转换为 BGR（OpenCV 默认格式）
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # 重组图片
            reorganized_image = reorganize_image(image_np, block_size=9, stride=3)

            # 将重组后的图像转换回 PIL 格式
            reorganized_pil = Image.fromarray(cv2.cvtColor(reorganized_image, cv2.COLOR_BGR2RGB))

            # 对重组后的图像进行像素化处理
            pixelated_image = pixelate_image(reorganized_pil, canvas_size=(32, 32))

            if pixelated_image:
                # 构建输出文件路径
                output_path = os.path.join(output_folder, f"pixelated_{filename}")
                # 保存结果
                pixelated_image.save(output_path, "PNG")
                print(f"图片处理完成，结果已保存为 {output_path}")
        except Exception as e:
            print(f"处理失败 {filename}: {str(e)}")

if __name__ == "__main__":
    input_dir = r"D:\python_project\image_sample\口袋妖怪一到五世代点阵像素图\口袋妖怪一到五世代点阵像素图\1口袋妖怪一世代"
    output_dir = r"D:\python_project\image_out\pokemon_01_1"
    process_folder(input_dir, output_dir)