from onnx_to_plugin import onnx2plugin

onnx2plugin(
	"/workspace/TensorRT-8.5.1.7/data/test_reduce_max.onnx",
	"output_reduce_max.onnx",
	node_names=["output_reduce_max.onnx"],
	node_types=["ReduceMax"],
        plugin_name_dict={"ReduceMax":"TPAT_ReduceMax"},
	dynamic_bs=False, # if True, this operator support dynamic batchsize
	min_bs=1,
	max_bs=256,
	opt_bs=256
	)
