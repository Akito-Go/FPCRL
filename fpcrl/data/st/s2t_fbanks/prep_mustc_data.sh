#!/bin/bash -v

set -e -o pipefail -x

export DATA=mustc
export PREP_PATH=/path/to/fairseq/examples/speech_to_text
export MUSTC_ROOT=/path/to/fpcrl/data/st/dataset/MuST-C
export SPM_INFO=/path/to/fpcrl/data/st/s2t_raw/spminfo/${DATA}
export SPM_DATA=/path/to/fpcrl/data/st/s2t_raw/spmdata/${DATA}

python ${PREP_PATH}/prep_mustc_data.py \
	--data-root ${MUSTC_ROOT} \
	--spm-info ${SPM_INFO} \
	--spm-data ${SPM_DATA} \
	--task st \
	--vocab-type unigram \
	--vocab-size 10000