预处理部分：
data_cleaning: 解析dicom并取出其中的图像序列转化为tiff（快去找份dicom数据）
downsample: 对tiff做降采样，对照3d_cleandata(我觉得写的不对，快重写！)
split_dataAndval: 划分测试集及标签，测试一下