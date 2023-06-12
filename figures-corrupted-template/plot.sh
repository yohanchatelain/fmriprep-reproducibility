python3 ../MRI-parse/parse_mri.py --reference ../outputs/template/pickle/rr ../outputs/template/pickle/rs --test one --mct-method fwe_bonferroni --template
mv one_mct_fwe_bonferroni__template.pdf template_fwe_bonferroni.pdf
mv one_pce__template.pdf template_pce.pdf
