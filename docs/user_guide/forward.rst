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
