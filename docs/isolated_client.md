# Isolated client

[Back to README](../README.md)

The standard installation is `pip install ayase`. This integration route is for
applications that need Ayase's full runtime in a separate Python environment.

Applications that already have their own PyTorch, NumPy, or media dependency
constraints can keep Ayase completely outside their environment:

```bash
pip install ayase --no-deps
```

```python
from ayase_client import AyasePipeline

pipeline = AyasePipeline(modules=["basic", "metadata", "motion"])
results = pipeline.run("./my_dataset")
pipeline.export("report.json")
pipeline.close()
```

This installs the same Ayase wheel without ML dependencies. On first use the
client creates a private full-runtime venv and starts a loopback worker. Media is
passed by local path. Regular `pip install ayase` remains unchanged.

Optional check: `ayase-client doctor`. To use an existing runtime, set
`AYASE_RUNTIME_PYTHON`.

This application-level worker is separate from the per-metric environments
Ayase manages automatically. See [managed backends](managed_backends.md) for
that infrastructure.
