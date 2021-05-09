## 一、项目说明
临近比赛提交时间结束才知道有这个AI ChatBot 创意赛【https://aistudio.baidu.com/aistudio/projectdetail/1902589】，匆忙申请好token，准备实现在微信上集成一些实用的计算机视觉小工具。由于剩余时间比较短，暂时只添加了OCR模块（chinese_ocr_db_crnn_mobile），基于VOC自定义训练的20分类目标检测模型后续会更新，以及使用paddlehub的进行自定义module也会在本文一并更新，请关注后续章节~

[GitHub repo](https://github.com/w-Bro/Wechaty-Paddlehub-CV-Tools)

## 二、wechaty准备
### 0. 环境说明与参考教程
```
padlocal服务器：VirtualBox 虚拟机 Centos7、Docker version 18.09.8, build 0dd43dd87f
本地运行服务器：Win10、python3.7

详细使用参见官方文档[https://python-wechaty.readthedocs.io/zh_CN/latest/introduction/use-padlocal-protocol/]
简单场景可参考以下步骤
```
### 2. 简单步骤一（padlocal服务器）：拉取镜像
```
docker pull wechaty/wechaty:latest
```

### 3. 简单步骤二（padlocal服务器）：启动padlocal服务
```sheel
export WECHATY_LOG="verbose"
export WECHATY_PUPPET="wechaty-puppet-wechat"
# 自定义填写
export WECHATY_PUPPET_SERVER_PORT="8080"
# 依据申请的填写
export WECHATY_TOKEN="puppet_padlocal_XXXX"
# 随机生成即可，参考下方代码块
export WECHATY_TOKEN="XXXX-c7ac2435b05d"

docker run -ti \
  --name wechaty_puppet_service_token_gateway \
  --rm \
  -e WECHATY_LOG \
  -e WECHATY_PUPPET \
  -e WECHATY_PUPPET_PADLOCAL_TOKEN \
  -e WECHATY_PUPPET_SERVER_PORT \
  -e WECHATY_TOKEN \
  -p "$WECHATY_PUPPET_SERVER_PORT:$WECHATY_PUPPET_SERVER_PORT" \
  wechaty/wechaty:latest
```
有看到以下输出，复制Online QR Code Image后面的链接，打开之后扫码登录，padlocal服务端就准备ok了
```
13:38:10 VERB StorageFile save() to /wechaty/XXXX-c7ac2435b05d.memory-card.json
13:38:10 INFO IoClient [5] https://login.weixin.qq.com/l/Acg3Az1aZQ==
Online QR Code Image: https://wechaty.js.org/qrcode/https%3A%2F%2Flogin.weixin.qq.com%2Fl%2FAcg3Az1aZQ%3D%3D
```

### 4. 简单步骤三（本地运行服务器）：运行demo
```python
import asyncio, os
from typing import List, Optional, Union

from wechaty_puppet import FileBox  # type: ignore

from wechaty import Wechaty, Contact
from wechaty.user import Message, Room

# 填步骤二生成的uuid v4
os.environ['WECHATY_PUPPET_SERVICE_TOKEN] = "XXXX-c7ac2435b05d"
# 填虚拟机（内网）地址
os.environ['WECHATY_PUPPET_SERVICE_ENDPOINT] = "192.168.1.101:8080"

class MyBot(Wechaty):

    async def on_message(self, msg: Message):
        """
        listen for message event
        """
        from_contact: Optional[Contact] = msg.talker()
        text = msg.text()
        room: Optional[Room] = msg.room()
        if text == 'ding':
            conversation: Union[
                Room, Contact] = from_contact if room is None else room
            await conversation.ready()
            await conversation.say('dong')
            file_box = FileBox.from_url(
                'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/'
                'u=1116676390,2305043183&fm=26&gp=0.jpg',
                name='ding-dong.jpg')
            await conversation.say(file_box)

asyncio.run(MyBot().start())
```
使用另外一个微信发送“ding”会收到回复“dong”则说明程序正常，wechaty配置完成，可以使用！

## 三、paddlehub准备
### 1. OCR(直接使用paddlehub预训练模型)
```
# 可参考官方示例【https://www.paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_crnn_mobile&en_category=TextRecognition】
# 我的封装代码 ocr_util.py
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
```

### 2. 目标检测（使用PaddleDetection训练VOC数据集以及自定义Paddlehub Module，待更新）
#### 2.1 VOC数据集准备，适配PaddleDetection格式

#### 2.2 使用PaddleDetection进行训练

#### 2.3 自定义Paddlehub Module

## 四、演示效果
![](https://ai-studio-static-online.cdn.bcebos.com/27fdf7b367de4932ac36217560af485dda54c94d43494479b9e87f0baf6b4e41)
![](https://ai-studio-static-online.cdn.bcebos.com/5d57b4bbd9454ef7b92ea1e99515d3389e887d8ea8b744cd93401eab063cdabc)



```python
# 克隆PaddleDetection仓库
# !git clone https://gitee.com/PaddlePaddle/PaddleDetection.git

# 编译安装paddledet 安装其他依赖
!cd PaddleDetection && python setup.py install && pip install -r requirements.txt

```


```python
# 测试
!cd PaddleDetection && python ppdet/modeling/tests/test_architectures.py
```


```python
# 在GPU上预测一张图片
! export CUDA_VISIBLE_DEVICES=0 && cd PaddleDetection && python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000570688.jpg
```


```python
# 生成uudiv4
import uuid
print(uuid.uuid4())
```

    8672500e-a7ea-43e1-a3a1-c0121bb493d0

