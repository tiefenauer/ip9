#!/usr/bin/env bash
./learning_curve.sh --run_id lc_cv_timit_dropouts \
                    --destination /media/D1/daniel.tiefenauer/_runs/ \
                    --train_files /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-train-rel.csv \
                    --valid_files /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-dev-rel.csv \
                    --lm /media/D1/daniel.tiefenauer/lm/timit_en/libri-timit-lm.klm \
                    --lm_vocab /media/D1/daniel.tiefenauer/lm/timit_en/libri-timit-lm.vocab \
                    --epochs 30 \
                    --valid_batches 30 \
                    --dropouts