#!/usr/bin/env bash
./learning_curve.sh -r lc_cv_timit -d /media/D1/daniel.tiefenauer/_runs/ -t /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-train-rel.csv -v /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-dev-rel.csv -l /media/D1/daniel.tiefenauer/lm/timit_en/libri-timit-lm.klm -a /media/D1/daniel.tiefenauer/lm/timit_en/libri-timit-lm.vocab -e 30 --valid_batches 30
