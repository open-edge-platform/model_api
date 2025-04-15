# Vision SDK

## Test building dependencies

You can build the package in docker by running from root model_api path.

```sh
docker build -t vision-api-build -f src/cpp/py_bindings/Dockerfile.ubuntu .
docker run --volume <path-to-data-folder>:/data -it vision-api-build bash
python src/cpp/py_bindings/run.py /data/classification_model_with_xai_head.xml /data/sheep.jpg
```

## Test dependencies to run python package

You can test that the python package runs by running from this path.

```sh
pip wheel .
docker build -t vision-api-test -f Dockerfile_test.ubuntu .
docker run --volume <path-to-data-folder>:/data -it vision-api-test bash
python run.py /data/classification_model_with_xai_head.xml /data/sheep.jpg
```
