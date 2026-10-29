import warnings

import pytest


@pytest.fixture(autouse=True)
def _silence_synthetic_optical_depth_warning():
    warnings.filterwarnings(
        "ignore",
        message="crtm_py optical depths are the synthetic",
        category=UserWarning,
    )
