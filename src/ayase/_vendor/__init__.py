"""Vendored dependencies with compatibility patches.

Currently vendored:
  clip/ — OpenAI CLIP (https://github.com/openai/CLIP). Vendored because the
          published package uses `from pkg_resources import packaging` which
          breaks on modern setuptools. Our copy uses `import packaging.version`.
          Replacement is installed in sys.modules by ayase._compat before any
          downstream consumer (pyiqa, ImageReward) tries to import clip.
"""
