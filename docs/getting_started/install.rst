安装
====

需要 Python 3.10 或更新版本。运行时依赖只有 NumPy。

.. code-block:: bash

   pip install -e ".[dev,docs]"
   pytest

从 git 装给别的项目用：

.. code-block:: text

   crtm-py @ git+https://github.com/IncubatorShokuhou/crtm-py

版本号是 0.1.1。仓库打上 ``v0.1.1`` 标签后可以把 URL 钉在 ``@v0.1.1``。导入名是 ``crtm_py``。文档：

.. code-block:: bash

   cd docs && make html

生成文件在 ``docs/_build/html``。官方 CRTM 系数不要装进源码树，见 :doc:`/user_guide/coefficients`。
