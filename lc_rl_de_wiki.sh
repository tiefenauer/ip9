#!/usr/bin/env bash
./learning_curve.sh --run_id lc_rl_de_wiki \
                    --destination /media/D1/daniel.tiefenauer/_runs/ \
                    --train_files /media/D1/daniel.tiefenauer/corpora/readylingua-de/readylingua-de-train.csv \
                    --valid_files /media/D1/daniel.tiefenauer/corpora/readylingua-de/readylingua-de-dev.csv \
                    --test_files /media/D1/daniel.tiefenauer/corpora/readylingua-de/readylingua-de-test.csv \
                    --lm /media/D1/daniel.tiefenauer/lm/wiki_de/wiki_de_5_gram_pruned.klm \
                    --lm_vocab /media/D1/daniel.tiefenauer/lm/wiki_de/wiki_de_5_gram_pruned.vocab \
                    --epochs 30 \
                    --language de
