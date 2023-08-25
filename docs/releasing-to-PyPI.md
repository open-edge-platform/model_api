```bash
py -m pip install --upgrade build
# Remove master omz_tools. Will use one from openvino-dev
sed -i '/omz_tools/d' model_api/python/requirements.txt
py -m build --sdist --wheel model_api/python/
py -m twine upload --username __token__ model_api/python/dist/*
git tag X.Y.Z
git push upstream X.Y.Z
```
Pull request to increment `version` in `setup.py` for the next release.
