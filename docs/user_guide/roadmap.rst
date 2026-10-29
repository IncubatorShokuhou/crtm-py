还没做的 CRTM 模块
==================

系数目录里有 ``abi_g16.SpcCoeff.bin`` 和 ODPS ``TauCoeff`` 时，前向模式按 CRTM v2.4 算。

已经对齐的部分：

* ODPS 预报量和层光学厚度，含水汽线的 compact OPTRAN。
* SpcCoeff 的通道 Planck 函数。
* ``CloudCoeff`` 红外消光、单次散射反照率、Legendre 系数，以及 delta 截断。
* 无散射时的镜面发射。``include_scattering=True`` 时走 ADA。
* 海面 Nalli（没有这份文件时用 Wu-Smith），陆地 NPOESS。

和 ``libcrtm.so``（JCSDA ``v2.4.0``，gfortran，``REAL64``）比，92 层晴空、天顶角 0° 和 30°，通道 7–16 最大差 0.059 K（通道 12）。同一条廓线在 700–900 hPa 放液水之后，通道 8–16 最大差 0.136 K（通道 11），通道 7 差 +0.80 K。逐通道数字在仓库 ``README``。

还没对齐的部分：

1. 通道 7 的有云亮温。ADA 的本征步对应 Fortran ``ASYMTX``。这里的相函数算子不对称，对称化之后其余通道反而更差，所以保留实部本征值。差集中在散射更强的 3.9 µm。
2. 通道 7 的白天太阳反射。
3. 冰、雪发射率系数。现在仍是常数。
4. 单独的 ODAS 气体系数。ABI G16 这份 TauCoeff 的主体是 ODPS。
5. 最大随机云重叠。``cloud_fraction`` 只在 CRTM 里把过小的云水清零；这里没有做这步。
6. NetCDF 系数。读的是 Fortran 顺序无格式文件，大端和小端都能读。
