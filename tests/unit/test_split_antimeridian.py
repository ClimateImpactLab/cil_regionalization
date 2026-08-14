"""Unit pins for the antimeridian-split criterion in the local backend.

The criterion is "polygon touches both +/-180 boundaries", NOT "bbox
wider than 180deg". A region wider than 180deg but not touching the
boundaries must NOT be split.
"""
from __future__ import annotations

import shapely
import shapely.geometry as sg

from cil_regionalization.backends.local import _split_at_antimeridian


class TestSplitCriterion:
    def test_genuinely_wide_non_crossing_region_not_split(self):
        """Bbox span 200deg but no contact with +/-180. Must stay intact.

        Coordinates -170..30 are >180deg wide. This shape is degenerate in
        canonical coords (such a polygon should already be wrapped at the
        dateline) but the split helper must not split it; it's the
        defensive guard the user asked for.
        """
        poly = sg.box(-170.0, -10.0, 30.0, 10.0)
        out = _split_at_antimeridian(poly)
        assert out == [poly]

    def test_fji_like_dateline_crosser_split_into_two(self):
        """A polygon that DOES touch both +/-180 gets split at lon=0."""
        west = sg.box(-179.8, -18.0, -178.0, -16.0)
        east = sg.box(178.0, -18.0, 179.8, -16.0)
        crosser = shapely.unary_union([west, east])
        halves = _split_at_antimeridian(crosser)
        assert len(halves) == 2
        # Each half lies entirely within one hemisphere.
        for h in halves:
            minx, _, maxx, _ = h.bounds
            assert minx >= -180.0 and maxx <= 180.0
            assert (maxx <= 0.0) or (minx >= 0.0)

    def test_polar_surrounding_region_split(self):
        """A region whose bbox touches both +/-180 (e.g. Antarctica) is
        split. The two halves cover disjoint ix ranges so the result
        stays correct even though the input doesn't 'cross' a dateline
        in the casual sense.
        """
        ata_like = sg.box(-180.0, -90.0, 180.0, -60.0)
        halves = _split_at_antimeridian(ata_like)
        assert len(halves) == 2

    def test_compact_mid_pacific_region_not_split(self):
        """A small region that doesn't touch +/-180 stays intact even if
        it's "near" the dateline.
        """
        poly = sg.box(170.0, -10.0, 175.0, 10.0)
        out = _split_at_antimeridian(poly)
        assert out == [poly]
