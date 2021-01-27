#!/bin/bash -x

if test -z $1 ; then
    echo "Training is skipped"
else
    echo "Training has started"
    python3 /src/main.py
fi
