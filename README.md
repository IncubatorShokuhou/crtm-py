# crtm-py

UPP v4.1 里 WRF-ARW 模拟卫星亮温用的 CRTM 前向过程，纯 Python / NumPy 重写。对齐 NOAA-EMC/UPP 标签 `dtc_post_v4.1.0` 的 `sorc/ncep_post.fd/CALRAD_WCLOUD_newcrtm.f`。传感器是 GOES-16/17 ABI 红外通道 7–16（`abi_g16`、`abi_g17`；UPP 在找不到 `abi_g16.SpcCoeff.bin` 时退回 `abi_gr`）。

给 [upp-py](https://github.com/IncubatorShokuhou/upp-py) 做可选依赖。安装名是 `crtm-py`，导入名是 `crtm_py`，版本 `0.1.0`。`upp-py[bt]` 以后可以写：

```text
crtm-py @ git+https://github.com/IncubatorShokuhou/crtm-py
```

打上 `v0.1.0` 标签之后，把 URL 末尾改成 `@v0.1.0`。发到 PyPI 之后则是 `crtm-py>=0.1`。

**English.** Pure-Python / NumPy port of the CRTM forward path UPP v4.1 uses for WRF-ARW ABI infrared brightness temperature. With official coefficients, ABI G16 channels 7–16 stay within 0.001 K of gfortran `libcrtm` (REAL64) on the CRTM clear-sky profile, including 60° zenith, land, liquid cloud and ice over liquid. Without coefficient files the package still runs a synthetic grey gas.

## 安装

```bash
pip install -e ".[dev,docs]"
pytest
cd docs && make html
```

运行时只依赖 NumPy。Python ≥ 3.10。

## 用法

气压、温度、比湿按 wrfout 的自下而上顺序（第 0 层靠近地面）。接口气压比层中气压多一层。

```python
import numpy as np
from crtm_py import brightness_temperature

p_int = np.array([1000e2, 700e2, 400e2, 200e2, 50e2])  # Pa
p_mid = np.array([850e2, 550e2, 300e2, 120e2])
t = np.array([290.0, 270.0, 250.0, 220.0])             # K
q = np.array([8e-3, 2e-3, 2e-4, 1e-5])                 # kg/kg

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
```

已经是 UPP 自上而下的数组时，设 `vertical_order="top_down"`。三维 `(n_levels, ny, nx)` 会展成格点再算，亮温形状是 `(ny, nx, n_channels)`。

格点天顶角可以不手填。给出经纬度后，`abi_g16` 用星下点 (0°N, 75.2°W)，`abi_g17` 用 (0°N, 137.2°W)，公式是 `GEO_ZENITH_ANGLE.f`。天顶角不在 `[0, 80]` 度内时该柱记为 NaN，对应 UPP 里不调用 `CRTM_Forward`、写出缺测的那些点。

水成物用 `Hydrometeors`，混合比单位 kg/kg，微物理方案号用 WRF 的 `mp_physics`。Thompson (8)、WSM6 (6)、Lin (2)、GFDL (11) 走 `EFFR`；Ferrier (5/85/95) 和 Zhao (98/99) 走 `CALRAD` 里内联的半径，不走 `EFFR`。

## 和 CRTM v2.4.0 的差别

系数放进 `CRTM_PY_COEFF_DIR` 之后，`abi_g16` 走 ODPS，不再走合成灰体。对拍用的是 JCSDA `v2.4.0` 用 gfortran、`REAL64` 编出来的 `libcrtm.so`。廓线是 CRTM 自带晴空测试的 92 层（`test_ClearSky/Load_Atm_Data.inc` 第一条）。海面是 Nalli，风速 5 m/s，水温 290 K，太阳天顶角 180°。

有云个例在 700–900 hPa 放液水 0.05 kg/m²，有效半径 10 µm，散射打开。冰云在 300–500 hPa，0.02 kg/m²，半径 20 µm。七种情形、十个通道，Python 减 Fortran 的最大绝对值是 0.0008 K。

| 情形 | 最大 \|Python − Fortran\| |
| --- | ---: |
| 晴空海面，天顶角 0° / 30° / 60° | 0.0002 / 0.0002 / 0.0008 K |
| 晴空陆地，NPOESS 类别 1，290 K | 0.00001 K 以内 |
| 液水云，天顶角 0° / 60° | 0.00001 K 以内 |
| 冰云叠在液水云上，天顶角 0° | 0.00001 K 以内 |

测试廓线最上面一层是 0.714 hPa。CRTM 在它上面用美国标准大气补到 0.005 hPa，再把补上的最下层平移，让温度和水汽、臭氧接到用户廓线。通道 12 看的是臭氧，这段平流层以前没有，天底就差 0.059 K；60° 路径更长，差到 0.122 K；上面再放冰云，差到 0.142 K。

预报量里还有一处。从第二层起，Fortran 的气压权重用层底气压 `Ref_Level_Pressure(k)`。之前写成了层顶。参考廓线上吸收物和参考值一样，这个比值是 1，错不出来。换到这条 ECMWF 廓线，臭氧的累积权重偏了，通道 12 的光学厚度跟着偏。干气体和水汽连续区不用这个臭氧权重，所以其他通道几乎看不出来。

送进 `ASYMTX` 的相函数矩阵本身不对称，和转置的差可以到 70 以上。`CRTM_Utility` 里 `ASYMTX` 的说明写的是：解一个事先知道本征值为实数的实不对称矩阵。它不是对称本征分解。以前说“CRTM 假设矩阵对称”是错的。

和 Fortran 对不上的第一处是云光学插值。`Get_Cloud_Opt_IR` 用 `Find_Random_Index` 加 `LPoly`，四个节点上两段二次多项式混在一起。之前这里用了相邻两点的线性插值，消光、单次散射反照率、Legendre 系数和 delta 截断一起偏了，相函数矩阵从这里开始分叉。求积点的数值是对的，但 Fortran 把最小的余弦放在前面；`Normalize_Phase` 从第一个流开始归一化，顺序反了，归一化之后的 `Pff`/`Pbb` 就不是同一个矩阵。这两处改完，同一层的 `HH`、本征值和本征向量与 Fortran 的 `ASYMTX` 一致。本征步按 `ASYMTX` 的实数算法写，不再对复数本征值取实部。

陆地发射率同样用这套四点二次插值。两点线性插值时，晴空陆地通道 7 差 0.13 K；改完之后是 0.001 K。

编对拍用的 `libcrtm` 时，JCSDA `v2.4.0` 的 `CMakeLists.txt` 第 6 行从

```text
project( crtm LANGUAGES Fortran )
```

改成

```text
project( crtm VERSION 2.4.0 LANGUAGES Fortran )
```

只加了版本号，给 ecbuild 用。计算选项没动。

没放系数时仍用合成灰体，`optical_depth_model` 还是 `synthetic_grey_gas+geometric_optics_cloud`。

| 环节 | 有系数时 |
| --- | --- |
| 气体光学厚度 | ODPS 第 2 组，水汽线用 compact OPTRAN（`OSignificance = 1`） |
| 云 | `CloudCoeff` 红外表，波数和有效半径用四点二次插值；散射打开时做 delta 截断和 ADA |
| 海面 / 陆地发射率 | Nalli IRSSEM（三线性）；NPOESS 陆地类别表（四点二次）。冰、雪仍是常数 |
| Planck | SpcCoeff 的 `Planck_C1/C2`、`Band_C1/C2` |
| 通道 7 白天太阳反射 | 没做 |
| ODAS 整条气体路径 | 没做。ABI G16 的 TauCoeff 是 ODPS，水汽线那一项已经是 OPTRAN |

## 系数

官方系数是 NOAA / JCSDA 的二进制，体积以 GB 计，不放进 git。CRTM REL-2.4.0 的 fix 包（约 3.4 GB）：

```text
https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.4.0.tgz
```

同目录还有更接近 UPP v4.1 年代的 `fix_REL-2.3.0_emc.tgz`。UPP v4.1 的 `ush/link_crtm_fix.sh` 默认只链接 `abi_gr`；`CALRAD` 在运行目录找不到 `abi_g16.SpcCoeff.bin` 时改用 `abi_gr`。

```bash
crtm-py fetch --list
crtm-py fetch --local-tarball /path/to/fix_REL-2.4.0.tgz
crtm-py fetch --yes          # 会下载约 3.4 GB，没有 --yes 会拒绝
```

解包只留 little-endian 的 ABI `SpcCoeff` / `TauCoeff`、`CloudCoeff.bin`、`AerosolCoeff.bin` 和名字里带 `Emis` 的文件，写到 `~/.cache/crtm-py/coefficients`。目录里同时有该传感器的 `SpcCoeff.bin` 和 ODPS `TauCoeff.bin` 时，前向模式会读它们。也可以用 `set_coeff_path` 或环境变量 `CRTM_PY_COEFF_DIR`（也认 `CRTM_COEFF_DIR`、`FIXCRTM`）。

对拍用的是 JEDI 2.4.0 系数包，大约 770 MB。文件名带 `.tar.gz`，内容是 POSIX tar，不是 gzip：

```text
https://bin.ssec.wisc.edu/pub/s4/CRTM/file/crtm_coefficients_jedi_release_2.4.0.tar.gz
```

`crtm-py fetch --list` 里的键是 `2.4.0-jedi`。

## 还没做

1. 通道 7 的白天太阳反射。
2. 冰、雪发射率系数。现在仍是常数。
3. 单独的 ODAS 气体路径。ABI G16 的 TauCoeff 是 ODPS。
4. 最大随机云重叠。`cloud_fraction` 在 CRTM 里还会把过小的云水清零，这里没有做这步。
5. 气溶胶、微波仪器。第 1 组微量气体预报量会直接报错。
6. NetCDF 系数。Fortran 顺序无格式文件的大端和小端都能读。
7. 和 `upp-py` 接线。这一版不改那个仓库。

CI 下载 JEDI 系数，用 `tests/test_crtm_reference.py` 里记下的 Fortran 亮温回归。不在 CI 里重编 `libcrtm`：本机已经用 JCSDA `v2.4.0`、gfortran、`REAL64` 跑过同一条廓线，数字写在测试里。ecbuild 要求 `project()` 带版本号，上游 `CMakeLists.txt` 要改一行才能编过。

## 文档

用户指南是中文。API 文档字符串是英文。

```bash
pip install -e ".[docs]"
cd docs && make html
```

## 许可

代码 MIT，见 `LICENSE`。CRTM / UPP 的来历和系数文件的归属见 `NOTICE`。
