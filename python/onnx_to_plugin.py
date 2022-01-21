##############################
# author : qianqiu
# email : qianqiu@tencent.com
# time : 2022.1.7
##############################
import argparse
import json
import onnx
import onnx_graphsurgeon as gs
from .cuda_kernel import CudaKernel
from .plugin_template_params import PluginTemplateParams
from .plugin_template import PluginTemplate
from .onnx_modified import OnnxModified



def generatePluginLibrary(input_model_path, nodes, plugin_name_dict=None):
    onnx_name_mapping_trt_plugin = {}
    trt_plugin_mapping_onnx_node = {}

    for node in nodes:
        tuning_name = node.name
        if plugin_name_dict is not None and tuning_name in plugin_name_dict.keys():
            plugin_name = plugin_name_dict[tuning_name]
        else:
            plugin_name = "tpat_" + str(node.name)
        assert (
            node.op != plugin_name
        ), "Please make sure your plugin name is different from op type in TensorRT, otherwise, the native kernel of tensorrt will be preferred for execution."
        cuda_kernel = CudaKernel(input_model_path, node, plugin_name)
        reusable_plugin = cuda_kernel.check_existing_plugins(
            trt_plugin_mapping_onnx_node
        )
        if reusable_plugin is None:
            print(
                "Couldn't find reusable plugin for node {}\nStart auto-tuning!".format(
                    cuda_kernel.tuning_name
                )
            )
            cuda_kernel.run()
            template_params = PluginTemplateParams(
                cuda_kernel, input_model_path, tuning_name
            )
            plugin_template = PluginTemplate(template_params)
            plugin_template.fill()
            onnx_name_mapping_trt_plugin[
                cuda_kernel.tuning_name
            ] = template_params.plugin_name
            trt_plugin_mapping_onnx_node[
                template_params.plugin_name
            ] = cuda_kernel._tuning_node
        else:
            print(
                "Find existing plugin {} which could be reused for node {}".format(
                    reusable_plugin, cuda_kernel.tuning_name
                )
            )
            onnx_name_mapping_trt_plugin[cuda_kernel.tuning_name] = reusable_plugin
    return onnx_name_mapping_trt_plugin


def onnx2plugin(
    input_model_path,
    output_model_path,
    node_names=None,
    node_types=None,
    plugin_name_dict=None,
):
    assert (
        node_names is not None or node_types is not None or plugin_name_dict is not None
    ), "Please input at least one of node name、node type and dict of plugin"
    try:
        input_onnx_model = onnx.load(input_model_path)
    except Exception as e:
        print("load onnx model : {} failed, Detail : {}".format(input_model_path, e))
        exit(1)
    input_model = gs.import_onnx(input_onnx_model)
    nodes = []
    if node_names is not None:
        for node_name in node_names:
            nodes.extend([node for node in input_model.nodes if node.name == node_name])
    if node_types is not None:
        for node_type in node_types:
            nodes.extend([node for node in input_model.nodes if node.op == node_type])
    if plugin_name_dict is not None:
        for one_plugin_name in plugin_name_dict.keys():
            nodes.extend(
                [node for node in input_model.nodes if node.name == one_plugin_name]
            )
    assert (
        len(nodes) != 0
    ), "Not get tuning node in onnx model, please check op name or onnx model"
    onnx_name_mapping_trt_plugin = generatePluginLibrary(
        input_model_path, nodes, plugin_name_dict
    )
    print("Onnx_name_mapping_trt_plugin: {}".format(onnx_name_mapping_trt_plugin))
    OnnxModified(
        input_model_path, output_model_path, nodes, onnx_name_mapping_trt_plugin
    )
    return onnx_name_mapping_trt_plugin.values()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_model_path",
        type=str,
        required=True,
        help="Please provide input onnx model path",
    )
    parser.add_argument(
        "-o",
        "--output_model_path",
        type=str,
        required=True,
        help="Please provide output onnx model path which used for tensorrt",
    )
    parser.add_argument(
        "-n",
        "--node_names",
        type=str,
        nargs="*",
        help="Please provide the operator name that needed to generate tensorrt-plugin",
    )
    parser.add_argument(
        "-t",
        "--node_types",
        type=str,
        nargs="*",
        help="Please provide the operator type that needed to generate tensorrt-plugin",
    )
    parser.add_argument(
        "-p",
        "--plugin_name_dict",
        type=str,
        help='Please provide the dict of op name and plugin name that \
            will be generated by TPAT, such as : {"op_name" : "plugin_name"}',
    )

    args = parser.parse_args()
    input_model_path = args.input_model_path
    output_model_path = args.output_model_path
    node_names, node_types, plugin_name_dict = None, None, None
    if args.node_names:
        node_names = args.node_names
    if args.node_types:
        node_types = args.node_types
    if args.plugin_name_dict:
        plugin_name_dict = json.loads(args.plugin_name_dict)
    onnx2plugin(
        input_model_path,
        output_model_path,
        node_names=node_names,
        node_types=node_types,
        plugin_name_dict=plugin_name_dict,
    )