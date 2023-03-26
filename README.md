<div align="center">
  <a href="https://v2.nonebot.dev/store"><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/nbp_logo.png" width="180" height="180" alt="NoneBotPluginLogo"></a>
  <br>
  <p><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/NoneBotPlugin.svg" width="240" alt="nonebot2_uocr_low"></p>
</div>

<div align="center">

# nonebot-plugin-ai-interviewer
</div>

## 介绍
- 由我仓库的另一个项目uocr拓展而来，基于keras mnist的手写数字识别，字母识别，提供三个训练好的模型，mnist,emnist-letter,emnist-byclass  
- 目前识别精度不高，mnist>byclass>letter,mnist只能识别数字，byclass只能识别字母且不区分大小写，class可以识别所有数字，字母类型但是精度不高
![demo](https://user-images.githubusercontent.com/69547456/227775526-6f549353-d3d1-4057-858a-2a43bdfaaef6.png)
## 安装  
（现在只有手动安装）
* 手动安装
  ```
  git clone https://github.com/canxin121/nonebot2_uocr_low.git
  ```  
  下载完成后在bot项目的pyproject.toml文件手动添加插件：  
  ```
  plugin_dirs = ["xxxxxx","xxxxxx",......,"下载完成的插件路径/nonebot-plugin-ai-interviewer"]  
  ``` 
## 使用方法

- 识别command /字符识别
```
#请输入模式(0:mnist,1:letter,2:byclass)和行数(0~4)，如
/字符识别02表示mnist两行

```
