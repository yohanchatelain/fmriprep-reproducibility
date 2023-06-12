#!/bin/bash

python3 ../MRI-parse/parse_mri.py --reference ../pickle/one/union/rr ../pickle/one/union/rs --test one --mct-method fwe_bonferroni --ratio
mv one_mct_fwe_bonferroni__ratio.pdf inter_mct_fwe_bonferroni.pdf
mv one_pce__ratio.pdf inter_pce.pdf
