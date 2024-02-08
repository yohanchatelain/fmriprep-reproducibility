#!/bin/bash

mkdir -p pickles/{loo,subjects,templates,versions}

parallel --jobs 16 --bar stabilitest single-test smri --config={} ::: configs/ieee/subjects/*.json
mv *subjects* pickles/subjects

parallel --jobs 16 --bar stabilitest single-test smri --config={} ::: configs/ieee/templates/*.json
mv *template* pickles/template

parallel --jobs 16 --bar stabilitest single-test smri --config={} ::: configs/ieee/versions/*.json
mv *versions* pickles/versions

parallel --jobs 16 --bar stabilitest cross-validation --model=loo smri --config={} ::: configs/loo/*.json
mv *loo* pickles/loo

