python3 ../MRI-parse/parse_mri.py --reference pickle/rr pickle/rs --test one --mct-method fwe_bonferroni --versions
mv one_mct_fwe_bonferroni_.pdf arch_fwe_bonferroni.pdf
mv one_pce_.pdf arch_pce.pdf
