#!/bin/bash -v

set -e -o pipefail -x

export CUDA_VISIBLE_DEVICES=0

export LANG=de
export MODE=train
export TYPE=s2t_hubert
export TASK=transfree
export DATA=covost
export PAIR=en-${LANG}
export BRAN=base_opr_fus_cst

export ROOT=/path/to/fpcrl
export GEN_SUBSET=test_st_${LANG}_en_raw
export SCRIPTS=/path/to/fairseq/scripts
export USER=${ROOT}/arch
export COVOST_ROOT=${ROOT}/data/st/dataset/CoVoST/cv-corpus-x.x/${LANG}
export SPM_INFO=${ROOT}/data/st/s2t_raw/spminfo/covost/${LANG}-en
export SAVE_DIR=${ROOT}/scripts/saveckp/${MODE}/${TYPE}/${TASK}/${METHOD}/${BRAN}

export BEAM=10
export LENPEN=1.4

#<<COMMENT
python ${SCRIPTS}/average_checkpoints.py \
	--inputs ${SAVE_DIR} \
	--num-epoch-checkpoints 10 \
	--output ${SAVE_DIR}/avg_last_10_checkpoint.pt
#COMMENT

fairseq-generate ${COVOST_ROOT} \
	--task speech_to_text \
	--user-dir ${USER} \
	--config-yaml ${SPM_INFO}/config_st_${LANG}_en.yaml \
	--gen-subset ${GEN_SUBSET} \
	--path ${SAVE_DIR}/avg_last_10_checkpoint.pt \
	--max-tokens 3000000 --max-source-positions 3000000 \
	--beam ${BEAM} --lenpen ${LENPEN} \
	--results-path ${SAVE_DIR}/scores/beam-${BEAM}.lp${LENPEN} \
	--ddp-backend no_c10d --distributed-world-size 1 \
	--scoring sacrebleu