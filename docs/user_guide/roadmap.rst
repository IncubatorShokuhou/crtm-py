还没做的 CRTM 模块
==================

已经按 UPP v4.1 源程序实现的部分：吸收物单位、水成物柱含量、``EFFR`` 与 Ferrier/Zhao 内联半径、植被类型和地表覆盖、静止卫星天顶角、Planck 函数、无散射发射求解。

亮温还不能当作 CRTM 结果，缺下面这些：

1. **气体光学厚度。** CRTM 2.3/2.4 的 ABI 系数是 ODPS 预报量。需要读取 ``TauCoeff``，按层的压力、温度、水汽、臭氧计算光学厚度。更老的 ODAS 只有在系数文件是 ODAS 时才需要。
2. **云光学性质。** ``CloudCoeff`` 给出消光系数、单次散射反照率、不对称因子，自变量是云种、有效半径和波段。现在用的几何光学把消光全部当成吸收，各红外通道共用同一个光学厚度。
3. **散射求解。** CRTM 的 ADA（adding-doubling）或 SOI。``ForwardModel.run(include_scattering=True)`` 会直接抛出 ``NotImplementedError``。
4. **云重叠。** UPP v4.1 不填 cloud fraction。CRTM 2.4 使用最大随机重叠。``Atmosphere.cloud_fraction`` 目前只把该层云光学厚度乘上这个分数。
5. **地表光学。** IRSSE 陆地发射率、红外水面与风速有关的反射、积雪发射率。通道 7（3.9 µm）白天还有太阳反射，需要 SpcCoeff 的太阳辐照度。
6. **光谱系数。** 用 SpcCoeff 的通道波数和 Planck 多项式替换现在的波段中心波数。NetCDF 比 Fortran 无格式二进制容易读，优先做 NetCDF。
7. **对比。** 同一条廓线上对比 ``unipost.exe`` 或 pyCRTM。在 1–3 完成之前，这种对比没有意义。

若 ODPS 移植长期无法和 CRTM 对齐，可以加一个可选的 Fortran 后端，调用 JCSDA CRTM 或 pyCRTM，输入输出仍用现在的 ``Atmosphere`` / ``Surface`` / ``Geometry``。那是后备，不是这一版的做法。
