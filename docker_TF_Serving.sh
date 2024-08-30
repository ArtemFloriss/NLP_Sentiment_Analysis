#!/bin/bash
docker stop tf_serving_sentiment_analysis
docker rm tf_serving_sentiment_analysis
docker run -p 8501:8501 --name=tf_serving_sentiment_analysis --mount type=bind,source=$(pwd)/tf_serving_model/sentiment_model,target=/models/sentiment_model -e MODEL_NAME=sentiment_model -t tensorflow/serving