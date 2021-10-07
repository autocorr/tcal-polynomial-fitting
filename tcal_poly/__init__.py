#!/usr/bin/env python3

from pathlib import Path


class ProjPaths:
    def __init__(self, extern, root, plot, poly):
        self.extern = Path(extern)
        self.root = Path(root)
        self.plot = self.root / plot
        self.poly = self.root / poly
        assert self.extern.exists()
        for path in (self.plot, self.poly):
            if not path.exists():
                path.mkdir()


PATHS = ProjPaths(
        "/lustre/aoc/sciops/fschinze/TCAL0009",
        "/lustre/aoc/sciops/bsvoboda/tcal_poly",
        "plots",
        "poly",
)


