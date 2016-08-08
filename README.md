# poseCaptioner_train

Train s2vt model using pose features. Code at `/home/gengshan/workJul/poseCaptioner_train`

## Procedure

### preprocess data and generate tsv, csv files
* run `python preprocess.py` to split data to train, val and test sets.
* .h5 input path and data output path are hard-coded in line 41, `lib/dataPreproLib.py`

### prepare ready-to-train .h5 files from generated sv files.
* run `prepareH5.ipynb`.

### traingng models with caffe
* e.g. `/data/gengshan/caffe/caffe/build/tools/caffe train -solver=./s2vt_solver.prototxt 2>train_log/log_8_2_16_50`
* The conventional blob dimensions for batches of image data are number N x channel K x height H x width W

### evaluate model
* run `eval_captioner.ipynb` to generate captions using trained model and dump html in `RESULTS_DIR`.

### wholestream evaluation
* compared to beam-search method, run `wholeStreamEval.ipynb` to get the same result as in training phase.

### show trainging statistics
* run `showLoss.ipynb` to plot training and validating loss.

### Lib files
* `dataPreproLib.py` defines methods for splitting datasets and forming .*sv files from a .h5 file.
* `dataLoader.py` defines a class **fc7FrameSequenceGenerator**, which reads in .csv and .tsv features and text.
* `hdf5_npstreamsequence_generator.py` defines a class **HDF5SequenceWriter**, which write the structured data in **fc7FrameSequenceGenerator** to ready-to-train .h5, and another class **SequenceGenerator** for constructing **fc7FrameSequenceGenerator**.
* `fileWriter.py` defines helper functions to write to .html files.
* `utils.py` defines function for searching outputs in testing phase.

### Prototxt files
* s2vt_solver.prototxt (training solver)
* s2vt.prototxt (training proto)
* s2vt.words_to_preds.deploy.prototxt (testing proto, for beam search)
* captioner.prototxt (testing proto, simulate training phase)

