# 拖放式音频分类器

这是一个 Windows Tkinter 小工具：给每张卡片绑定一个真实文件夹，然后把 WAV 从桌面或资源管理器拖到卡片上，文件会自动移动到对应目录。

## 安装

~~~powershell
python -m pip install tkinterdnd2
~~~

## 使用

1. 运行 <code>python folder_drop_router.pyw</code>。
2. 给 A/B 卡片设置路径。
3. 使用右上角“新增文件夹”添加 C/D。
4. 建议映射：
   - A → 能用（声音正常）
   - B → 能用（声音较高）
   - C → 暂留（背景音或其他）
   - D → 不留
5. 将一个或多个文件拖到对应卡片。

同名文件不会覆盖，而会自动命名为 <code>文件名 (1).wav</code>。程序只移动普通文件，不移动整个目录。

