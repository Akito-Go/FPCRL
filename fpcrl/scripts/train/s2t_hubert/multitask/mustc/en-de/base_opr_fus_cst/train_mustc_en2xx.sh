#!/bin/bash -v

set -e -o pipefail -x

export CUDA_VISIBLE_DEVICES=0

export LANG=de
export MODE=train
export TYPE=s2t_hubert
export TASK=multitask
export DATA=mustc
export PAIR=en-${LANG}
export BRAN=base_opr_fus_cst

export ARCH=s2t_hubert_mtl_opr_post
export CRITERION=label_smoothed_cross_entropy_mtl_opr
export FUS_WP=15000
export CST_WP=15000

export ROOT=/path/to/fpcrl
export USER=${ROOT}/arch
export MUSTC_ROOT=${ROOT}/data/st/dataset/MuST-C/en-${LANG}
export PREMODEL=${ROOT}/arch/models/pretrain
export MTMODEL=${ROOT}/scripts/saveckp/pretrain/s2t_mustc/en-${LANG}/mtl_mt/avg_last_10_checkpoint.pt
export SPM_INFO=${ROOT}/data/st/s2t_raw/spminfo/mustc/en-${LANG}
export SAVE_DIR=${ROOT}/scripts/saveckp/${MODE}/${TYPE}/${TASK}/${DATA}/${PAIR}/${BRAN}
export LOG_DIR=${ROOT}/scripts/${MODE}/${TYPE}/${TASK}/${DATA}/${PAIR}/${BRAN}/${BRAN}_log

mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}

fairseq-train ${MUSTC_ROOT} \
	--task speech_and_text \
	--user-dir ${USER} \
	--arch ${ARCH} --save-dir ${SAVE_DIR} \
	--use-jsd --use-fus --fus-warmup ${FUS_WP} --use-cst --cst-warmup ${CST_WP} \
	--hubert-model-path ${PREMODEL}/hubert_base_ls960.pt --mt-model-path ${MTMODEL} \
	--config-yaml ${SPM_INFO}/config_st.yaml --train-subset train_st_raw --valid-subset dev_st_raw \
	--max-tokens 3000000 --max-source-positions 900000 --max-target-positions 1024 --batch-size 42 \
	--max-epoch 96 --max-update 60000 --num-workers 4 \
	--log-interval 10 --keep-last-epochs 10 --patience 10 \
	--criterion ${CRITERION} --label-smoothing 0.1 --report-accuracy \
	--layernorm-embedding --optimizer adam --adam-betas "(0.9, 0.98)" \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 25000 \
	--clip-norm 0.0 --seed 1 --update-freq 8 \
	--skip-invalid-size-inputs-valid-test \
	--fp16 --memory-efficient-fp16 \
	--ddp-backend no_c10d --distributed-world-size 1 \
	2>&1 | tee ${LOG_DIR}/train.log