import urllib.request

import numpy as np
from cv2 import imdecode
from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.typing import T_State

from ._recognize import recognize

digit = on_command("字符识别")


@digit.handle()
async def digitrecog(bot: Bot, event: Event, state: T_State):
    global img, fullimg
    msg = event.get_message()
    num_row = msg[0].data['text'][-1]
    mode = msg[0].data['text'][-2]
    if ('0' < num_row <= '4') and ('0' <= mode <= '2'):
        for seg in msg:
            if seg.type == 'image':
                fullimg = seg.data['url']
                resp = urllib.request.urlopen(fullimg)
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = imdecode(arr, -1)
        msg = recognize(img, int(num_row), mode)
    else:
        msg = '请输入模式(0:mnist,1:letter,2:byclass)和行数(0~4)，如/字符识别02表示mnist两行'
    await digit.finish(msg)
