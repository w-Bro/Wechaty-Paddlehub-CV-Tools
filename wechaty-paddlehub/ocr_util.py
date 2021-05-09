#!/usr/bin/env python
import cv2

import paddlehub as hub

# 加载移动端预训练模型
ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")


def ocr_text(img_path):
    """
    OCR识别
    :param img_path:
    :return:
    """
    # 读取测试文件夹test.txt中的照片路径
    np_images =[cv2.imread(img_path)]

    results = ocr.recognize_text(
                        images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                        use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                        visualization=False,       # 是否将识别结果保存为图片文件；
                        box_thresh=0.5,           # 检测文本框置信度的阈值；
                        text_thresh=0.5,          # 识别中文文本置信度的阈值；
                        )
    text_results = []
    for result in results:
        data = result['data']
        # save_path = result['save_path']
        text = "文本：置信度\n"
        for infomation in data:
            # print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ', infomation['text_box_position'])
            text += f"{infomation['text']}：{round(infomation['confidence'], 2)}\n"
        text_results.append(text.strip())

    return text_results

if __name__ == '__main__':
    print(ocr_text("image/1125858391207285698.jpg"))