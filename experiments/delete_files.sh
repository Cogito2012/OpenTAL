#!/bin/bash

cd ../models/thumos14/$1
find . ! -path '*tensorboard*' ! -path "*latest*" ! -path '*25*' -delete
cd ../../../