# save()

## Function

Quantization parameter configuration class, which saves the quantized weights and related parameters by encapsulating the quantization algorithm through the calibrator class.

Note: Because there is a deserialization risk during the process of saving quantization parameters, it is necessary to mitigate the risk by setting the permissions of the saved quantization result folder to 750, the quantization weight file permissions to 400, and the quantization weight description file permissions to 600 during the saving process.

## Prototype

```python
calibrator.save(output_path="")
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| output_path | Input | The path to save the quantized weights and related parameters. | Required.<br>Data type: string. |

## Sample

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()    # Load according to the actual model situation.
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
calibrator.run() 
calibrator.save("./quant_model.ckpt")
```
