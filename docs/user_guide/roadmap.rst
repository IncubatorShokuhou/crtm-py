还没做的 CRTM 模块
==================

系数目录里有 ``abi_g16.SpcCoeff.bin`` 和 ODPS ``TauCoeff`` 时，前向模式按 CRTM v2.4 算。

已经对齐的部分：

* ODPS 预报量和层光学厚度，含水汽线的 compact OPTRAN。
* SpcCoeff 的通道 Planck 函数。
* ``CloudCoeff`` 红外消光、单次散射反照率、Legendre 系数，以及 delta 截断。
* 无散射时的镜面发射。``include_scattering=True`` 时走 ADA。
* 海面 Nalli（没有这份文件时用 Wu-Smith），陆地 NPOESS。

和 ``libcrtm.so``（JCSDA ``v2.4.0``，gfortran，``REAL64``）比，晴空海面 0°/30°/60°、晴空陆地、液水云 0°/60°、冰云叠液水云，通道 7–16 最大差 0.0008 K。用户顶低于 0.005 hPa 时，上面按美国标准大气补层。臭氧预报量的气压权重用层底气压。逐通道数字在仓库 ``README``。

``ASYMTX`` 解的是实不对称矩阵，本征值事先知道是实数。送进去的 ``HH`` 并不对称。云系数用四点二次插值，求积余弦从小到大排列。逐柱路径按 ``ASYMTX`` 的实数算法算。批量路径用 ``numpy.linalg.eig``，一块里只对有云的层做这个分解。

编这份 ``libcrtm`` 时，上游 ``CMakeLists.txt`` 第 6 行的 ``project( crtm LANGUAGES Fortran )`` 改成 ``project( crtm VERSION 2.4.0 LANGUAGES Fortran )``。只加版本号，计算没有改。

还没对齐的部分：

1. 通道 7 的白天太阳反射。
2. 冰、雪发射率系数。现在仍是常数。
3. 单独的 ODAS 气体系数。ABI G16 这份 TauCoeff 的主体是 ODPS。
4. 最大随机云重叠。``cloud_fraction`` 只在 CRTM 里把过小的云水清零；这里没有做这步。
5. NetCDF 系数。读的是 Fortran 顺序无格式文件，大端和小端都能读。
