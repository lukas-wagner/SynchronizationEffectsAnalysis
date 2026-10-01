"""Shared pytest fixtures and path setup for the SyncroEffects test suite."""

import os
import sys
import json
import copy

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulation"))
sys.path.insert(0, os.path.join(ROOT, "optimization"))


@pytest.fixture
def cfg():
    """A fresh copy of the project config.json for each test."""
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as fh:
        return copy.deepcopy(json.load(fh))
