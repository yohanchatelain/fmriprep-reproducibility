python3 ../MRI-parse/parse_mri.py --reference ../pickle/exclude/union/rr ../pickle/exclude/union/rs --test exclude --mct-method fwe_bonferroni 
mv exclude_mct_fwe_bonferroni.pdf loo_fwe_bonferroni.pdf
mv exclude_pce.pdf loo_pce.pdf
