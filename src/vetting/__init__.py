#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os

from .centroiding import centroid_test

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

__all__ = ["centroid_test"]
