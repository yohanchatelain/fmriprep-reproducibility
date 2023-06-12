python3 ../MRI-parse/parse_mri.py --reference ../pickle/versions/union/rr ../pickle/versions/union/rs --test one --mct-method fwe_bonferroni --versions
mv one_mct_fwe_bonferroni_.pdf versions_fwe_bonferroni.pdf
mv one_pce_.pdf versions_pce.pdf
