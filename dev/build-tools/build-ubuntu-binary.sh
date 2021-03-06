#!/bin/sh
# These scripts are here for those interested in attempting a binary build.
# It NOT RECOMMENDED, please use build-venv.sh

cd ..
rm -r build
rm -r dist
rm BEAGLES.spec
pip3 install pyinstaller
PYINSTALLER=$(find / -name pyinstaller 2> /dev/null)

pip3 install -r requirements/requirements.txt
make all
${PYINSTALLER} --hidden-import=xml \
            --hidden-import=xml.etree \
            --hidden-import=xml.etree.ElementTree \
            --hidden-import=lxml.etree \
            -r libs/cython_utils/cy_yolo_findboxes.so \
            -r libs/cython_utils/cy_yolo2_findboxes.so \
            -r libs/cython_utils/nms.so \
            --add-data ./data:data \
            --icon=resources/icons/app.icns \
            -n BEAGLES slgrSuite.py \
            -D -p ./libs -p ./


FOLDER=$(git describe --abbrev=0 --tags)
FOLDER="linux_"$FOLDER
rm -rf "$FOLDER"
mkdir "$FOLDER"


# Picking up the slack for pyinstaller
SITEPKG=$(python3 -c "import site; print(site.USER_SITE)")
TFCONTRIB=dist/BEAGLES/tensorflow/contrib
cp_tf_contrib () {
    for var in "$@"
        do
            mkdir -p ${TFCONTRIB}/"$var"/python/ops/
            find ${SITEPKG}/tensorflow/contrib/"$var"/python/ops/ \( -name "*.so" \) -exec cp -r {} dist/BEAGLES/tensorflow/contrib/"$var"/python/ops/ \;
            find ${SITEPKG}/tensorflow/contrib/"$var"/ \( -name "*.so" \) -exec cp -r {} dist/BEAGLES/tensorflow/contrib/"$var"/ \;
        done
}

cp_tf_contrib all_reduce autograph batching bayesflow bigtable boosted_trees checkpoint cloud cluster_resolver cmake coder compiler constrained_optimization copy_graph crf cudnn_rnn data decision_trees deprecated distribute distributions eager estimator factorization feature_column ffmpeg framework fused_conv gan graph_editor grid_rnn hadoop hooks ignite image input_pipeline integrate kafka keras kernel_methods kinesis labeled_tensor layers learn legacy_seq2seq libsvm linear_optimizer lite lookup losses memory_stats meta_graph_transform metrics mixed_precision model_pruning nearest_neighbor nn opt optimizer_v2 periodic_resample predictor proto quantization quantize rate receptive_field recurrent reduce_slice_ops remote_fused_graph resampler rnn rpc saved_model seq2seq session_bundle signal slim solvers sparsemax specs staging stateless stat_summarizer summary tensorboard tensor_forest tensorrt testing text tfprof timeseries tpu training util
cp libs/cython_utils/nms.cpython-* dist/BEAGLES/libs/cython_utils # Figure out why Cython won't copy this during setup

# Creating a zipfile
cp -rf dist/BEAGLES $FOLDER
zip "$FOLDER.zip" -r $FOLDER
