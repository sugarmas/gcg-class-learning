# CS5185 Index

香港城市大学（东莞）Multimedia Technologies and Applications。

主线：media ↔ systems ↔ trade-offs。媒体流之间要协同一致，才有连贯的 QoE。

## 总览

- [[多媒体 知识地图]] —— 四条主线、依赖链、反复出现的数学工具、易混对照、复习顺序（**先看这篇**）

## 笔记

**概论与图像**
- [[数字图像表示]] —— 像素/分辨率/色深、bit-depth expansion、gamma、直方图、dithering、PPI/DPI、插值、PSNR/SSIM、RGB/YUV/YIQ

**数字化：采样与量化**
- [[采样与混叠]] —— 采样定理、奈奎斯特频率、混叠、低通滤波、空间/时间采样
- [[量化]] —— 均匀居中/非居中、非均匀、误差与信噪比

**压缩**
- [[视频压缩基础]] —— 四种冗余、RLE、字典编码、熵、Huffman、变换、帧类型、混合编码器
- [[JPEG 图像压缩]] —— DCT、量化表、DC 的 DPCM、zigzag、AC 的 RLC、色度子采样、伪影
- [[PNG 与 DEFLATE]] —— 预测滤波、Sub/Up/Paeth、LZ77、Huffman
- [[学习式图像压缩]] —— VAE、加噪近似量化、hyperprior、HiFiC、保真 vs 感知

**系统与传输**
- [[服务质量 QoS]] —— 时延、抖动、缓冲区、拥塞管理、自适应码率、协议

## 习题

- [[CS5185 题型与答案]] —— 五次 tutorial 按**七类题型**重排，**答案全部直接放出**（含 week5 官方答案对照）

## 材料位置

`C:\Users\Lenovo\Desktop\研究生\多媒体\`，按周分五个子文件夹：

| 文件夹 | 内容 |
|---|---|
| 第一周 | Week1.pdf（概论、图形动画、具身 AI）|
| 第二周 | Image Representation 讲义 + Questions |
| 第三周 | Sampling and QoS 讲义 + Quantization tutorial |
| 第四周 | VC_basic 讲义 + Tutorial Questions |
| 第五周 | Image Compression 讲义 + JPEG/PNG tutorial（题目 + 官方答案）|

全部 PDF 均可提取文字，无扫描件。

## 相关

- [[多媒体 知识地图]]
- [[CS5185 题型与答案]]
- [[Coursework Index]]
