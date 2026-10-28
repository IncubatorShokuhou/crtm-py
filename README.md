# crtm-py

UPP v4.1 里 WRF-ARW 模拟卫星亮温用的 CRTM 前向过程，纯 Python / NumPy 重写。对齐 NOAA-EMC/UPP 标签 `dtc_post_v4.1.0` 的 `sorc/ncep_post.fd/CALRAD_WCLOUD_newcrtm.f`。传感器是 GOES-16/17 ABI 红外通道 7–16（`abi_g16`、`abi_g17`；UPP 在找不到 `abi_g16.SpcCoeff.bin` 时退回 `abi_gr`）。

给 [upp-py](https://github.com/IncubatorShokuhou/upp-py) 做可选依赖。安装名是 `crtm-py`，导入名是 `crtm_py`，版本 `0.1.0`。`upp-py[bt]` 以后可以写：

```text
crtm-py @ git+https://github.com/IncubatorShokuhou/crtm-py
```

打上 `v0.1.0` 标签之后，把 URL 末尾改成 `@v0.1.0`。发到 PyPI 之后则是 `crtm-py>=0.1`。

**English.** Pure-Python / NumPy port of the CRTM forward pieces UPP v4.1 uses for WRF-ARW ABI infrared brightness temperature. Profile conversion, effective radius and geostationary zenith follow `CALRAD_WCLOUD_newcrtm.f`. Gas optical depth and cloud optical properties do not yet use official CRTM coefficients, so the temperatures are not CRTM output.

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

## 这一版算出来的亮温

| 环节 | 状态 |
| --- | --- |
| 吸收物单位：水汽 g/kg 混合比，臭氧 ppmv，缺测臭氧记 1e-10 | 按 `CALRAD` |
| 水成物柱含量 `q * dp / g`，对流云 `icu_physics=2` | 按 `CALRAD` |
| `EFFR`（含 WSM6 冰半径顶在 250 µm、Thompson 霰用 `RG=g` 这两处源程序行为） | 按 `CALRAD` |
| 植被类型表、地表覆盖、皮肤温度 | 按 `CALRAD` |
| 静止卫星天顶角 | 按 `GEO_ZENITH_ANGLE.f` |
| Planck 函数，波数用 ABI 波段中心 | 公式按 CRTM；不是 SpcCoeff 通道积分 |
| 等温层、平面平行、朗伯漫射度 5/3 的发射求解 | 无散射极限 |
| 气体光学厚度 | **合成灰体**，不是 ODPS / ODAS |
| 云消光 | **几何光学** `3 Qext / (4 ρ re)`，`Qext=2`，单次散射反照率为 0。不是 CloudCoeff |
| 地表发射率 | 水面 0.98、陆地 0.97、冰 0.98、雪 0.99，按覆盖加权。不是 IRSSE / IRwater |
| ABI 通道 7 的太阳反射 | 没有 |

所以 10 个红外通道都能返回亮温，数值**不能**和 `unipost.exe` 或 pyCRTM 对表。结果对象的 `optical_depth_model` 是 `synthetic_grey_gas+geometric_optics_cloud`，`notes` 里写了同样的限制。

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

解包只留 little-endian 的 ABI `SpcCoeff` / `TauCoeff`、`CloudCoeff.bin`、`AerosolCoeff.bin` 和名字里带 `Emis` 的文件，写到 `~/.cache/crtm-py/coefficients`。这一版**不读**这些文件。目录也可以用 `set_coeff_path` 或环境变量 `CRTM_PY_COEFF_DIR`（也认 `CRTM_COEFF_DIR`、`FIXCRTM`）指过去，前向结果的 `notes` 会记一笔“路径已设置，读取尚未实现”。

## 还没做

1. ODPS 预报量光学厚度（`TauCoeff`），以及旧的 ODAS。
2. `CloudCoeff` 查表：消光、单次散射反照率、不对称因子，随有效半径和波段变化。
3. 带散射的 ADA / SOI。最大随机云重叠（CRTM 2.4 的 cloud fraction）只预留了数组，光学厚度目前按层分数相乘。
4. IRSSE 陆地发射率和红外水面 Fresnel。通道 7 的太阳反射需要 `SpcCoeff` 里的太阳辐照度。
5. SpcCoeff 的 NetCDF / Fortran 二进制读取，以及多色 Planck 系数。
6. 气溶胶、微波仪器。
7. 和 `unipost.exe` 或 pyCRTM 的同廓线对比。等 1–3 落地再做。

纯 Python 重写仍然是目标。若 ODPS 移植长期对不上 CRTM，再考虑可选的 Fortran 后端（pyCRTM 或对 JCSDA CRTM 的薄封装）作为额外入口，不替换现在的数组接口。这一版没有链接 Fortran。

## 文档

用户指南是中文。API 文档字符串是英文。

```bash
pip install -e ".[docs]"
cd docs && make html
```

## 许可

代码 MIT，见 `LICENSE`。CRTM / UPP 的来历和系数文件的归属见 `NOTICE`。
