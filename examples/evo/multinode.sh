#!/bin/bash

echo "=================== process 0 output ==================="
CUDA_VISIBLE_DEVICES=2,3 python test_multinode.py 1 2 > /tmp/toy_1.out &
CUDA_VISIBLE_DEVICES=0,1 python test_multinode.py 0 2

wait
echo

echo "=================== process 1 output ==================="
cat /tmp/toy_1.out
echo
