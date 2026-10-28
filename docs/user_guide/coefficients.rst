系数文件
========

SpcCoeff、TauCoeff、CloudCoeff 和发射率查找表是 NOAA / JCSDA 的数据，不提交到这个仓库。当前前向模式也不读取它们。

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

也可以把已有目录指给进程：

.. code-block:: python

   from crtm_py import set_coeff_path

   set_coeff_path("/path/to/crtm/fix")

环境变量按顺序认 ``CRTM_PY_COEFF_DIR``、``CRTM_COEFF_DIR``、``FIXCRTM``。路径会记在前向结果的 ``notes`` 里。ODPS 和 CloudCoeff 的读取还没写，所以设了路径也不会改变亮温。
