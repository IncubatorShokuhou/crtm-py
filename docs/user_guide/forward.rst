算亮温
======

下面这列是自下而上的，和 wrfout 一样。接口气压 5 个数，层中气压 4 个数。

.. code-block:: python

   import numpy as np
   from crtm_py import brightness_temperature

   p_int = np.array([1000e2, 700e2, 400e2, 200e2, 50e2])
   p_mid = np.array([850e2, 550e2, 300e2, 120e2])
   t = np.array([290.0, 270.0, 250.0, 220.0])
   q = np.array([8e-3, 2e-3, 2e-4, 1e-5])

   result = brightness_temperature(
       p_int, p_mid, t, q,
       sensor="abi_g16",
       channels=(8, 14),
       vertical_order="bottom_up",
       sensor_zenith=0.0,
       skin_temperature=292.0,
   )
   print(result.brightness_temperature)
   print(result.optical_depth_model)

``result.brightness_temperature`` 单位是 K，顺序跟 ``channels`` 一致。没放系数时 ``optical_depth_model`` 是 ``synthetic_grey_gas+geometric_optics_cloud``。放了 SpcCoeff 和 ODPS TauCoeff 之后，缺省是 ``odps+specular_emission``；``include_scattering=True`` 时是 ``odps+cloudcoeff+ada``。

数组
----

* 一维：单柱。亮温形状 ``(n_channels,)``。
* 二维 ``(n_profiles, n_levels)``：一批柱，层在最后一维。
* 三维 ``(n_levels, ny, nx)``：和 ``upp-py`` 的 WRF 状态一样。亮温形状 ``(ny, nx, n_channels)``。

``vertical_order="bottom_up"`` 时第 0 层是地面。``"top_down"`` 时第 0 层是模式顶，和 UPP / CRTM 一致。排反了会报错。

``abi_g16`` 的星下点是 (0°N, 75.2°W)，``abi_g17`` 是 (0°N, 137.2°W)。传入 ``latitude`` 和 ``longitude``（东经为正）时用 ``GEO_ZENITH_ANGLE.f`` 的球面公式。``abi_gr`` 保持天顶角 0，除非单独传入 ``sensor_zenith``。天顶角不在 0 到 80 度之间，该柱为 NaN。

水成物和地表
------------

.. code-block:: python

   from crtm_py import Hydrometeors

   hydro = Hydrometeors(qqw=q * 0)  # 形状与温度相同，单位 kg/kg
   result = brightness_temperature(
       p_int, p_mid, t, q,
       hydrometeors=hydro,
       imp_physics=8,
       sensor_zenith=0.0,
       skin_temperature=292.0,
   )

``imp_physics`` 用 WRF 的微物理选项。8 是 Thompson，6 是 WSM6，2 是 Lin，11 是 GFDL，5/85/95 是 Ferrier，98/99 是 Zhao。Ferrier 和 Zhao 的半径用 ``CALRAD`` 内联公式：Zhao 云水 10 µm、云冰 50 µm；Ferrier 云冰 75 µm。其余方案调用 ``EFFR``，返回值已经是半径。

没有 ``sea_mask`` 或 ``landmask`` 时，整柱当成陆地，皮肤温度用 ``skin_temperature``，缺省则用最下层气温。``landmask`` 是 WRF 的（1 为陆地）。``sea_mask`` 是 UPP 的 ``sm``（大于 0.1 为水）。植被类型 ``ivegsrc``：0 USGS，1 IGBP，2 旧 GFS。

臭氧省略时按 WRF 分支处理，吸收物记 1e-10 ppmv。传入的臭氧是 kg/kg，内部乘 604229 换成 ppmv。

没放系数时，每一层按等温层算，天顶方向的光学厚度乘 ``1/cos(zenith)``，地表向下的热辐射用漫射度割线 5/3。没有散射。

放了系数之后，缺省仍是无散射的镜面发射。``include_scattering=True`` 改用 ADA，云的消光、单次散射反照率和 Legendre 系数来自 ``CloudCoeff``。海面发射率读 Nalli（没有这份文件时用 Wu-Smith），陆地读 NPOESS。冰、雪仍是常数。通道 7 的白天太阳反射两条路径都没有。

``include_clouds=False`` 时两种路径都不加云。

速度
----

有系数时按 ``chunk_size`` 分块，缺省 256 列。``n_jobs`` 缺省 1。大于 1 时用进程池，并把 ``OMP_NUM_THREADS`` 设成 1。

云水不超过 1e-6 kg/m² 的层不查 ``CloudCoeff``。单次散射反照率不超过 1e-10 的层用对角发射算子。整柱没有散射时走无散射的发射公式。要散射的层叠在一起做 ``numpy.linalg.eig``。这些 ``HH`` 的本征值是实数，排好序之后和 ``ASYMTX`` 大约差 1e-12，本征向量差一个比例。加层公式不看这个比例。天顶角落在高斯节点上时，那一柱仍用逐柱的 ``ASYMTX``。

改前基准有两处。Carr Fire wrfout，199×199 列、79 层、ABI 7–16、散射开：main 上大约 139–169 ms/列/核，8 核全场墙钟 11–14 分钟，``cProfile`` 热点在 ``ada_radiance``。逐柱路径在 CRTM 晴空测试廓线上大约是晴空 10 ms/列、液水云 64 ms/列。

下面这张表是改后的，机器是 4 核，Python 3.12，NumPy 1.26，``OMP_NUM_THREADS=1``，``chunk_size=256``。廓线重复自那条 92 层测试柱。液水在 700–900 hPa。偶数柱那一列只给偶数柱放云。复现用 ``benchmarks/bench_abi.py --full``。

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22

   * - 列数
     - 晴空
     - 每列液水
     - 偶数柱液水
   * - 1
     - 0.004 秒（3.62 ms/列）
     - 0.012 秒（12.2 ms/列）
     - 0.012 秒（12.1 ms/列）
   * - 100
     - 0.021 秒（0.208 ms/列）
     - 0.216 秒（2.16 ms/列）
     - 0.115 秒（1.15 ms/列）
   * - 1000
     - 0.274 秒（0.274 ms/列）
     - 2.16 秒（2.16 ms/列）
     - 1.18 秒（1.18 ms/列）
   * - 39601
     - 11.2 秒（0.283 ms/列）
     - 82.5 秒（2.08 ms/列）
     - 46.9 秒（1.19 ms/列）

``n_jobs=4`` 时，39601 列每列都有液水，墙钟 23.2 秒。这台机器是 4 核。

单线程、每列都有液水的全场大约一分半。Carr Fire 里晴空柱和有云柱混在一起，单线程落在晴空的 11 秒和全液水的 83 秒之间。那次 wrfout 是 79 层，这张表补上标准大气之后大约 100 层。
