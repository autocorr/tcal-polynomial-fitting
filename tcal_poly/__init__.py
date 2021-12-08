#!/usr/bin/env python3

from pathlib import Path


FRANK_DIR = Path("/lustre/aoc/sciops/fschinze")
BRIAN_DIR = Path("/lustre/aoc/sciops/bsvoboda")
PROJ_DIR  = Path("/lustre/aoc/projects/vlacal")

class ProjPaths:
    def __init__(self, extern, root, plot, data):
        self.extern = Path(extern)
        self.root = Path(root)
        self.plot = self.root / plot
        self.data = self.root / data
        assert self.extern.exists()
        if not self.plot.exists():
            self.plot.mkdir()


PATHS = ProjPaths(
        extern=PROJ_DIR/"TCAL0009",
        root=BRIAN_DIR/"tcal_poly",
        plot="plots",
        data="poly",
)


