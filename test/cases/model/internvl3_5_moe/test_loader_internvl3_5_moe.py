from msmodelslim.model.internvl3_5_moe.loader import InternVL3_5MoeAdapterLoader


def test_adapter_class_path_given_loader_when_called_then_return_expected():
    assert (
        InternVL3_5MoeAdapterLoader.ADAPTER_CLASS_PATH
        == "msmodelslim.model.internvl3_5_moe.model_adapter:InternVL3_5MoeModelAdapter"
    )


def test_loader_instantiation_given_default_when_called_then_succeed():
    loader = InternVL3_5MoeAdapterLoader()
    assert loader.ADAPTER_CLASS_PATH == "msmodelslim.model.internvl3_5_moe.model_adapter:InternVL3_5MoeModelAdapter"
