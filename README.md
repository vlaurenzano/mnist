# mnist
Using the MNIST data set with python and sklearn to classify handwritten digits. 

## Configuration
Configure via config.ini. DEFAULT arguments can be overriden per algorithm. 

## Run 
Run by passing argument "fit", "model", or "predict" to main.py or Docker.
Must run with fit with the same parameters before running predict. 
"model" will run k-folds validation and print score. 

```
usage: main.py [-h] [--algo ALGO] mode

Classify and predict digits using the mnist dataset

positional arguments:
  mode         the mode to run in: fit, model or predict

optional arguments:
  -h, --help   show this help message and exit
  --algo ALGO  which algorithm to use: RandomForest, KNN
 ``` 
 
## With docker-compose

Install with:

``` docker-compose build ```

Run with:

``` docker-compose mnist fit ```

``` docker-compose mnist predict ```



