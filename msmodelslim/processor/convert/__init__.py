#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Data-free IR transform processors for offline convert.
"""

from msmodelslim.processor.convert.base import BaseConvertProcessor
from msmodelslim.processor.convert.registry import register_convert_processors

__all__ = ["BaseConvertProcessor", "register_convert_processors"]
