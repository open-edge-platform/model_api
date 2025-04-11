# Vision SDK


## Test building dependencies:

``` sh
docker build -t vision-api-build -f ./Dockerfile.ubuntu .
```

## Test dependencies to run python package

You can test that the python package runs by running from this path:

``` sh
pip wheel .
docker build -t vision-api-test -f Dockerfile_test.ubuntu . 
docker run --volume <path-to-data-folder>:/data -it vision-api-test bash 
python scratch.py /data/classification_model_with_xai_head.xml /data/sheep.jpg
```
