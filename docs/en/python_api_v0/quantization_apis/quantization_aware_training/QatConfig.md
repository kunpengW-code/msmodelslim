# QatConfig

## Function

Quantization parameter configuration class, used to save the parameters configured during the quantization process.

## Prototype

```python
QatConfig(w_bit=8, a_bit=8, a_sym=False, amp_num=0, steps=1, ema=0.99, is_forward=False, ignore_head_tail_node=False, disable_names=None, has_init_quant=False, quant_mode=True, grad_scale=0.0, compressed_model_checkpoint=None, opset_version=11, save_params=False, input_names=None, output_names=None, save_onnx_name=None)
```

## Parameters

| Parameter Name | Input | Description | Constraints |
|---------| ------ |----------------------------------------------------------|-------------------------------------------------|
| w_bit | Input | Weight quantization bit. | Optional.<br>Data Type: int.<br>Default is 8, modification is not supported. |
| a_bit | Input | Activation layer quantization bit. | Optional.<br>Data Type: int.<br>Default is 8, modification is not supported. |
| a_sym | Input | Whether to use symmetric quantization for activation values. | Optional.<br>Data Type: bool.<br>Default is False. |
| amp_num | Input | Number of automatic fallback layers.<br>When accuracy drops significantly, you can increase the number of fallback layers. It is recommended to prioritize fallback of 1 to 3 layers. If accuracy recovery is not significant, then increase the number of fallback layers. | Optional.<br>Data Type: int.<br>Value Range is [0,10], default is 0. Inputs such as 1, 2, 3 are allowed. |
| steps | Input | Number of steps for automatic fallback. | Optional.<br>Data Type: int.<br>Default is 1, value range is greater than or equal to 1. |
| ema | Input | Parameter in the Adam optimizer, the exponential moving average metric. | Optional.<br>Data Type: float.<br>Value Range is [0.1,1.0], default is 0.99. |
| is_forward | Input | Whether to process the forward pass with reference to mmdetection. | Optional.<br>Data Type: bool.<br>Default is False. |
| ignore_head_tail_node | Input | Whether to ignore the first and last layers and not quantize them. | Optional.<br>Data Type: bool.<br>Default is False. |
| disable_names | Input | Names of nodes to be excluded from quantization, i.e., the names of quantization layers for manual fallback.<br>If the accuracy is poor, you can select quantization layers for fallback. | Optional.<br>Data Type: list[str].<br>Default is None. |
| has_init_quant | Input | Whether the model has undergone quantization initialization. | Optional.<br>Data Type: bool.<br>Default is False. |
| quant_mode | Input | Whether to enable quantization mode. | Optional.<br>Data Type: bool.<br>Default is True. |
| grad_scale | Input | Gradient compensation strength. | Optional.<br>Data Type: float.<br>Default is 0.0, recommended configuration is 0.001. |
| compressed_model_checkpoint | Input | The weight file and its path for the pseudo-quantized model saved when exporting the ONNX model. | Optional.<br>Data Type: string.<br>Default is None. |
| opset_version | Input | Version number when exporting the ONNX model. The corresponding ONNX version must be installed in advance. | Optional.<br>Data Type: int.<br>Optional values are 11 and 13, default is 11. |
| save_params | Input | Whether to save quantization-related parameters as .npy files during export. | Optional.<br>Data Type: bool.<br>Default is False. |
| input_names | Input | Input names for ONNX. | Optional.<br>Data Type: list[str]<br>Default is None. |
| output_names | Input | Output names for ONNX. | Optional.<br>Data Type: list[str]<br>Default is None. |
| save_onnx_name | Input | Pseudo-quantized model weights. | Optional.<br>Data Type: str.<br>Default is None. |

## Sample

```python
from msmodelslim.pytorch.quant.qat_tools import QatConfig
quant_config = QatConfig(grad_scale=0.001)
```
