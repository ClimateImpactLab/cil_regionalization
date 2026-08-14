"""WeightSpec dispatch helpers."""
from __future__ import annotations

import pytest

from cil_regionalization.config import WeightConfig
from cil_regionalization.weights import WeightSpec, from_config_list


def test_is_area_flag():
    a = WeightSpec.from_config(WeightConfig(name="area"))
    p = WeightSpec.from_config(WeightConfig(name="pop", raster="/tmp/pop.tif"))
    assert a.is_area is True
    assert p.is_area is False


def test_require_raster_returns_path():
    p = WeightSpec.from_config(WeightConfig(name="pop", raster="/tmp/pop.tif"))
    assert p.require_raster() == "/tmp/pop.tif"


def test_require_raster_raises_for_area():
    a = WeightSpec.from_config(WeightConfig(name="area"))
    with pytest.raises(ValueError, match="weights.area.raster"):
        a.require_raster()


def test_require_table_returns_id():
    p = WeightSpec.from_config(WeightConfig(name="pop", table="ci.gpw.pop"))
    assert p.require_table() == "ci.gpw.pop"


def test_require_table_raises_for_area():
    a = WeightSpec.from_config(WeightConfig(name="area"))
    with pytest.raises(ValueError, match="weights.area.table"):
        a.require_table()


def test_from_config_list_preserves_order():
    cfgs = [
        WeightConfig(name="pop", raster="/tmp/p.tif"),
        WeightConfig(name="area"),
        WeightConfig(name="crop", raster="/tmp/c.tif"),
    ]
    specs = from_config_list(cfgs)
    assert [s.name for s in specs] == ["pop", "area", "crop"]
