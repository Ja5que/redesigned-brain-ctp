python环境暂时设置3.9
天哪python的包管理简直就是狗屎，傻X动态语言，我恨我自己，为什么我在写python，暂时修了下requirement（通过除去所有torch安装命令会在window安装的包）

conda create -n vistest2 python=3.9  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r .\requirements_new.txt  

预处理部分：
data_cleaning: 解析dicom并取出其中的图像序列转化为tiff（快去找份dicom数据）
downsample: 对tiff做降采样，对照3d_cleandata(我觉得写的不对，快重写！)
split_dataAndval: 划分测试集及标签，测试一下

input:15*256*256
mask:30*512*512