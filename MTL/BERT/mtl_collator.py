import torch
from typing import List, Dict, Any

class MTLBatchCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict:
        if not batch:
            raise ValueError("Batch is empty!")
        collated_batch = {
            "smiles": [],
            #"fingerprint": [],
            "labels": [],
            "smiles_variants": [],
        }
        for item in batch:
            collated_batch["smiles"].append(item["smiles"])
            #collated_batch["fingerprint"].append(item["fingerprint"])
            collated_batch["labels"].append(item["labels"])
            smiles_variants = item.get("smiles_variants", None)
            collated_batch["smiles_variants"].append(smiles_variants if smiles_variants is not None else [])
        #collated_batch["fingerprint"] = torch.stack(collated_batch["fingerprint"])
        collated_batch["labels"] = torch.stack(collated_batch["labels"])
        return collated_batch