from typing import Dict, Any

class ModelPartition:

    def __init__(self, model):
        self.model = model
    
    def get_partitions(self) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def get_partition(self, id_part) -> Any:
        raise NotImplementedError()
    
    def set_partition(self, id_part, model_part) -> None:
        raise NotImplementedError()
    
    def merge_partition(self, id_part, model_part) -> None:
        raise NotImplementedError()


class LayerPartition(ModelPartition):
    
    def get_partitions(self):
        partitions : Dict = {}
        for name, _ in self.model.named_parameters():
            layer_name : str  = name.split('.')[0]
            partitions[layer_name] = getattr(self.model, layer_name)
        return partitions
    
    def get_partition(self, id_part):
        return getattr(self.model, id_part)