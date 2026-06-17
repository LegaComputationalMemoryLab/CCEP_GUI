# Contributing

1. Do not include patient data, identifiable clinical metadata, images, or reports in issues, pull requests, commits, screenshots, or test fixtures.
2. Use synthetic examples only.
3. Run the compile check before opening a pull request:

```bash
python -m py_compile src/ieeg_ccep_analyzer/app.py
python -m pytest
```

4. Describe the clinical or research use case for each feature request.
5. For changes affecting analysis outputs, include a clear before/after explanation and note any impact on manuscript claims.
