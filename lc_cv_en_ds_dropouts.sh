#!/usr/bin/env bash
gpu=$1
if [[ ${gpu} = '' ]]; then
    echo "Enter GPU # to use for training"
    read gpu
fi
echo "using GPU #${gpu} for all training runs!"

./learning_curve.sh --run_id lc_cv_en_ds_dropouts \
                    --destination /media/D1/daniel.tiefenauer/_runs/ \
                    --train_files /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-train-rel.csv \
                    --valid_files /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-dev-rel.csv \
                    --test_files /media/D1/daniel.tiefenauer/corpora/cv/cv-valid-test-rel.csv \
                    --lm /media/D1/daniel.tiefenauer/lm/ds_en/lm.binary \
                    --lm_vocab /media/D1/daniel.tiefenauer/lm/ds_en/lm.vocab \
                    --epochs 30 \
                    --valid_batches 30 \
                    --dropouts \
                    --gpu ${gpu}
