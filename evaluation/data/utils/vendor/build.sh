#!/bin/bash

git clone https://github.com/tree-sitter/tree-sitter-c
cd tree-sitter-c
python setup.py build; python setup.py install

cd ..

git clone https://github.com/tree-sitter/tree-sitter-cpp
cd tree-sitter-cpp
python setup.py build; python setup.py install
