# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lightweight config loader for GroundingDINO model configs.

Reads a plain Python ``*.py`` config file (a flat set of module-level literal
assignments) into an attribute-accessible object, matching the small subset of
the upstream ``SLConfig`` API that model construction relies on:
``SLConfig.fromfile(path)`` followed by ``args.<name>`` attribute reads/writes.

This implementation intentionally depends only on the standard library so the
vendored package installs with no extra third-party dependencies.
"""

import os.path as osp


class ConfigDict(dict):
    """A ``dict`` that also supports attribute-style access to its keys."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


def _load_py_config(filename):
    """Execute a ``*.py`` config file and return its module-level names."""
    filename = osp.abspath(osp.expanduser(filename))
    if not osp.isfile(filename):
        raise FileNotFoundError("Config file not found: {}".format(filename))
    if not filename.endswith(".py"):
        raise IOError("Only .py config files are supported, got: {}".format(filename))

    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()

    namespace = {"__file__": filename}
    exec(compile(source, filename, "exec"), namespace)

    cfg = {}
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        # skip imported modules / callables that are not plain config values
        if getattr(value, "__module__", None) == "builtins" and callable(value):
            continue
        cfg[key] = value
    return cfg


class SLConfig:
    """Minimal attribute-accessible view over a config dict.

    Only the members exercised by GroundingDINO's ``build_model`` path are
    provided: construction from a ``*.py`` file and attribute get/set.
    """

    def __init__(self, cfg_dict=None):
        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict or {}))

    @staticmethod
    def fromfile(filename):
        return SLConfig(_load_py_config(filename))

    def __getattr__(self, name):
        # only reached when ``name`` is not a normal instance attribute
        try:
            return self._cfg_dict[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._cfg_dict[name] = value

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __setitem__(self, name, value):
        self._cfg_dict[name] = value

    def __contains__(self, name):
        return name in self._cfg_dict

    def __repr__(self):
        return "SLConfig({})".format(dict(self._cfg_dict))
