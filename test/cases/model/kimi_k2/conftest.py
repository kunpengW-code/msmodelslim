#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest
from transformers import PretrainedConfig


class _DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=0, **kwargs)
        self.hidden_size = 64
        self.vocab_size = 128
        self.rms_norm_eps = 1e-6
        self.num_hidden_layers = 3


@pytest.fixture
def dummy_config():
    return _DummyConfig()
