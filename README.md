# NanoTrack_RK3588_python

基于瑞芯微RK3588 NPU的NanoTrack跟踪算法，可运行于RK3588开发板，可达120FPS.


## dependence

```
	numpy
	opencv
	rknn_toolkit_lite2 == 1.3 
```

RKNN3588对应的rknn_toolkit_lite2官方开发库以及开发文档请参考[rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)


## demo

模型转换需先使用rknn-toolkit2转为.rknn格式

```
	python3 main.py
```

- video_name 为目标视频地址
- init_rect 为初始检测bbox


## reference

[rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)  
[SiamTracker](https://github.com/HonglinChu/SiamTrackers)
