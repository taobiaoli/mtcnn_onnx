'''
Created by YKX taobiaoli on 2020.11.09
'''
import onnx
# 
def modify_input_shape(model):
    '''
    Helper function to change onnx model input shape used to dynamic input shape.
    PNet dynamic input: 1*3*?*?. only modify h and w.
    parameter model: loaded onnx model
    return:changed onnx model.
    '''
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    return model

def onnx_add_output(model,node_name):
    '''
    Helper function to change onnx model struct used to run in onnxruntime enviroment and output each node tensor.
    parameter model: loaded onnx model
    return: changed onnx model.
    '''
    #model = onnx.load(model_path)
    added_outputs = []
    # from quantize node list choose node and find the outputs from model node
    for model_node in model.graph.node:
        if model_node.name == node_name:
            intermediate_node_value_info = onnx.helper.ValueInfoProto()
            intermediate_node_value_info.name = model_node.output[0]
            added_outputs.append(intermediate_node_value_info)
    model.graph.output.extend(added_outputs)
    return model
def modify_reshape_batch(model,op_name):
    '''
    Helper function to change onnx model input shape used to dynamic input shape.
    The shape inference is wrong when Gemm add Prelu. PRelu modified shape 1*128 from 128*1*1 .
    parameter model: loaded onnx model
    return:changed onnx model.
    '''
    input_maps = {}
    init_maps = {}
    keys = []
    for inp in model.graph.input:
        input_maps[inp.name] = inp
        keys.append(inp.name)
    for init in model.graph.initializer:
        init_maps[init.name] = init
    for key in keys:
        if op_name in key:
            inp = input_maps[key]
            dim_value = inp.type.tensor_type.shape.dim[0].dim_value
            print('dim_value',dim_value)
            new_shape = [1,dim_value]
            model.graph.input.remove(inp)
            new_inp = onnx.helper.make_tensor_value_info(inp.name,onnx.TensorProto.FLOAT,new_shape)
            model.graph.input.extend([new_inp])
            init = init_maps[key]
            new_init = onnx.helper.make_tensor(init.name,onnx.TensorProto.FLOAT,new_shape,init.float_data)
            model.graph.initializer.remove(init)
            model.graph.initializer.extend([new_init])
    return model

if __name__ == '__main__':
    pnet = onnx.load('det1.onnx')
    pnet_shape = modify_input_shape(pnet)
    pnet_shape_output = onnx_add_output(pnet_shape,'conv4-1')
    onnx.save(pnet_shape_output,'optimaizer_pnet.onnx')
    rnet = onnx.load('det2.onnx')
    rnet_prelu_shape = modify_reshape_batch(rnet,'prelu4')
    onnx.save(rnet_prelu_shape,'optimaizer_rnet.onnx')

    onet = onnx.load('det3.onnx')
    onet_prelu_shape = modify_reshape_batch(onet,'prelu5')
    onnx.save(onet_prelu_shape,'optimaizer_onet.onnx')
