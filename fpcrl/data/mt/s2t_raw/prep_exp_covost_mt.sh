#!/bin/bash -v

set -e -o pipefail -x

export SRC_LANG=de
export TGT_LANG=en
export SPM_MODEL=/path/to/fpcrl/data/st/s2t_raw/spminfo/covost/de-en/spm_unigram10000_st_de_en.model
export SPM_DICT=/path/to/fpcrl/data/st/s2t_raw/spminfo/covost/de-en/spm_unigram10000_st_de_en.txt

export TEXT_PATH_TRAIN=/home/jlou/FRCRL/fpcrl/data/mt/dataset/WMT/wmt16/de-en
export TEXT_PATH_DEV=/home/jlou/FRCRL/fpcrl/data/mt/dataset/WMT/wmt16/de-en

export DATA=covost
export PAIR=${SRC_LANG}-${TGT_LANG}
export TEXT_SPM=/path/to/fpcrl/data/mt/s2t_raw/spminfo/${DATA}/${PAIR}
export PREP=/path/to/fpcrl/data/mt/s2t_raw

mkdir -p ${TEXT_SPM}

# train data
python3 ${PREP}/apply_spm.py --model ${SPM_MODEL} --input-file ${TEXT_PATH_TRAIN}/train.${SRC_LANG} --output-file ${TEXT_SPM}/train.spm.exp.${SRC_LANG} --add_lang_tag ${SRC_LANG}
python3 ${PREP}/apply_spm.py --model ${SPM_MODEL} --input-file ${TEXT_PATH_TRAIN}/train.${TGT_LANG} --output-file ${TEXT_SPM}/train.spm.exp.${TGT_LANG} --add_lang_tag ${TGT_LANG}

# dev data
python3 ${PREP}/apply_spm.py --model ${SPM_MODEL} --input-file ${TEXT_PATH_DEV}/dev.${SRC_LANG} --output-file ${TEXT_SPM}/dev.spm.exp.${SRC_LANG} --add_lang_tag ${SRC_LANG}
python3 ${PREP}/apply_spm.py --model ${SPM_MODEL} --input-file ${TEXT_PATH_DEV}/dev.${TGT_LANG} --output-file ${TEXT_SPM}/dev.spm.exp.${TGT_LANG} --add_lang_tag ${TGT_LANG}

fairseq-preprocess \
    --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} \
    --trainpref ${TEXT_SPM}/train.spm.exp --validpref ${TEXT_SPM}/dev.spm.exp \
    --destdir ${TEXT_SPM}/exp_mt --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict ${SPM_DICT} --tgtdict ${SPM_DICT} \
    --workers 100