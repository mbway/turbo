#!/usr/bin/env python3

import turbo.utils

import pytest


def test_remap():
    # al => range a lower
    al, au = (5, 99)
    bl, bu = (0.5, 1.0)
    assert turbo.utils.remap(23, (al, au), (bl, bu)) == pytest.approx(bl + (bu-bl)/(au-al) * (23-al))
    assert turbo.utils.remap(0.6, (bl, bu), (al, au)) == pytest.approx(al + (au-al)/(bu-bl) * (0.6-bl))
