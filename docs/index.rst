crtm-py
=======

UPP v4.1 WRF-ARW 亮温（ABI 红外）的纯 Python 前向过程。系数目录里有 SpcCoeff 和 ODPS TauCoeff 时按 CRTM v2.4 算。没有这些文件时用合成灰体，亮温不能和 ``unipost.exe`` 对表。

.. toctree::
   :maxdepth: 2

   getting_started/install
   user_guide/forward
   user_guide/coefficients
   user_guide/roadmap
   api
