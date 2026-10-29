系数文件
========

SpcCoeff、TauCoeff、CloudCoeff 和发射率查找表是 NOAA / JCSDA 的数据，不提交到这个仓库。前向模式在目录里找到该传感器的 ``SpcCoeff.bin`` 和 ODPS ``TauCoeff.bin`` 时会读取。读的是 Fortran 顺序无格式文件，大端和小端都可以。NetCDF 还没读。

CRTM REL-2.4.0 的 fix 压缩包大约 3.4 GB：

.. code-block:: text

   https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.4.0.tgz

同一目录下的 ``fix_REL-2.3.0_emc.tgz`` 更接近 UPP v4.1 链接的 CRTM 年代。UPP 标签 ``dtc_post_v4.1.0`` 的 ``ush/link_crtm_fix.sh`` 默认只链接 ``abi_gr``。``CALRAD`` 若在运行目录找不到 ``abi_g16.SpcCoeff.bin``，传感器 id 会改成 ``abi_gr``。

列出已知地址：

.. code-block:: bash

   crtm-py fetch --list

本地已有压缩包时解包，不访问网络：

.. code-block:: bash

   crtm-py fetch --local-tarball /path/to/fix_REL-2.4.0.tgz

需要下载时必须加 ``--yes``。解包结果在 ``~/.cache/crtm-py/coefficients``，只保留 little-endian 的 ABI 文件、``CloudCoeff.bin``、``AerosolCoeff.bin`` 以及文件名中含 ``Emis`` 的系数。``crtm_py_manifest.json`` 记录实际写出的文件名。

对拍用的 JEDI 2.4.0 包大约 770 MB，键名 ``2.4.0-jedi``。文件名带 ``.tar.gz``，内容是 POSIX tar：

.. code-block:: text

   https://bin.ssec.wisc.edu/pub/s4/CRTM/file/crtm_coefficients_jedi_release_2.4.0.tar.gz

也可以把已有目录指给进程：

.. code-block:: python

   from crtm_py import set_coeff_path

   set_coeff_path("/path/to/crtm/fix")

环境变量按顺序认 ``CRTM_PY_COEFF_DIR``、``CRTM_COEFF_DIR``、``FIXCRTM``。SpcCoeff 或 ODPS TauCoeff 缺一件时仍走合成灰体，``notes`` 里会写明。找到这两份文件后，气体光学厚度用 ODPS，云用 ``CloudCoeff``，海面用 Nalli，陆地用 NPOESS。
