from unittest.mock import Mock

import pytest

from salmon.errors import UploadError
from salmon.uploader.preassumptions import print_preassumptions


@pytest.mark.parametrize("spectrals", [(), (0,), (1,)])
@pytest.mark.parametrize("after", [True, False])
@pytest.mark.parametrize("urls", [(), (1,)])
def test_lossy_master_requires_generated_or_supplied_spectrals(spectrals, after, urls):
    args = (Mock(), "album", None, "WEB", True, spectrals, None, after, urls)
    if spectrals == (0,) and not after and not urls:
        with pytest.raises(UploadError, match="without spectrals"):
            print_preassumptions(*args)
    else:
        print_preassumptions(*args)
