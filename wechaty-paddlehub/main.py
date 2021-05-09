#!/usr/bin/env python

import asyncio
import os
from typing import Optional, Union

from wechaty import Wechaty, Contact
from wechaty.user import Message, Room
from wechaty_puppet import FileBox  # type: ignore
from ocr_util import ocr_text

os.environ['WECHATY_PUPPET_SERVICE_TOKEN'] = "XXX-XXX-a116-c7ac2435b05d"
os.environ["WECHATY_PUPPET_SERVICE_ENDPOINT"] = '192.168.0.186:8080'

task_code = -1


class MyBot(Wechaty):

    async def on_message(self, msg: Message):
        """
        listen for message event
        """
        from_contact: Optional[Contact] = msg.talker()
        text = msg.text()
        room: Optional[Room] = msg.room()
        conversation: Union[
            Room, Contact] = from_contact if room is None else room

        global task_code

        if text == 'help':
            await conversation.ready()
            await conversation.say(f'{conversation.name}，你好，我是AI-CV小工具\n'
                                   f'你可以回复编码使用对应功能喔\n'
                                   f'1: OCR识别\n'
                                   f'2: 目标检测\n')
            # file_box = FileBox.from_url(
            #     'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/'
            #     'u=1116676390,2305043183&fm=26&gp=0.jpg',
            #     name='ding-dong.jpg')
            # await conversation.say(file_box)

        if text == '1':
            task_code = 1
            await conversation.ready()
            await conversation.say(f"请发送需要进行OCR的图片")

        # 如果收到的message是一张图片
        if msg.type() == Message.Type.MESSAGE_TYPE_IMAGE:
            # 将Message转换为FileBox
            file_box_2 = await msg.to_file_box()
            # 获取图片名
            img_name = file_box_2.name
            # 图片保存的路径
            img_path = './image/' + img_name

            print(task_code)
            # 将图片保存为本地文件
            await file_box_2.to_file(file_path=img_path)

            conversation: Union[
                Room, Contact] = from_contact if room is None else room
            await conversation.ready()
            await conversation.say("已收到图片，正在处理，请稍候...")

            if task_code == 1:
                text = ocr_text(img_path)
                await conversation.say("OCR识别成功！识别结果如下：")
                await conversation.say(text[0])

                # 处理完成后置空code
                task_code = -1


asyncio.run(MyBot().start())