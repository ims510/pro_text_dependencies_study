import lal

tbproc = lal.io.treebank_processor()
err=tbproc.init("vectors.txt", "num_crossings.csv")
if err == lal.io.treebank_error_type.no_error: 
    tbproc.clear_features()
    tbproc.add_feature(lal.io.treebank_feature.num_crossings)
    err=tbproc.process()
print(err)