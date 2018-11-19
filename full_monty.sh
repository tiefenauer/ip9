#!/usr/bin/env bash
# do the full monty, i.e. create learning curves for all possible combinations

echo "Enter corpus to train on ('cv'=CommonVoice, English, 'rl'=ReadyLingua, German): "
read corpus

echo "Enter GPU # to use for training: "
read gpu

echo "training on ${corpus} corpus using GPU ${gpu} for all training runs!"

if [[ ${corpus} = 'cv' ]]; then
    ./lc_cv_en_ds.sh ${gpu}
#    ./lc_cv_en_ds_adam.sh ${gpu}
    ./lc_cv_en_ds_dropouts.sh ${gpu}
#    ./lc_cv_en_ds_dropouts_adam.sh ${gpu}

#    ./lc_cv_en_timit.sh ${gpu}
#    ./lc_cv_en_timit_adam.sh ${gpu}
#    ./lc_cv_en_timit_dropouts.sh ${gpu}
#    ./lc_cv_en_timit_dropouts_adam.sh ${gpu}
elif [[ ${corpus} = 'rl' ]]; then
    ./lc_rl_de_wiki.sh ${gpu}
#    ./lc_rl_de_wiki_adam.sh ${gpu}
    ./lc_rl_de_wiki_dropouts.sh ${gpu}
#    ./lc_rl_de_wiki_dropouts_adam.sh ${gpu}
    ./lc_rl_de_wiki_synth.sh ${gpu}
#    ./lc_rl_de_wiki_synth_adam.sh ${gpu}
    ./lc_rl_de_wiki_synth_dropouts.sh ${gpu}
#    ./lc_rl_de_wiki_synth_dropouts_adam.sh ${gpu}
else
    echo "invalid value $corpus for corpus"
fi