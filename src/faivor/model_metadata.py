import json
from typing import List, Dict, Any, Optional


class ModelMetadata:
    """
    Class to store metadata information for a ML model, following FAIRmodels syntax.
    """

    def __init__(self, metadata_json: Dict[str, Any]):
        self.metadata: Dict[str, Any] = metadata_json
        self.inputs: List[Dict[str, Any]] = self._parse_inputs()
        self.output: str = self._parse_output()
        self.output_label: Optional[str] = metadata_json.get("Outcome label", {}).get("@value")

        general_info = self.metadata.get("General Model Information", {})
        self.docker_image: Optional[str] = general_info.get("FAIRmodels image name", {}).get("@value")
        self.model_name: Optional[str] = general_info.get("Title", {}).get("@value")
        self.description: Optional[str] = general_info.get("Editor Note", {}).get("@value")
        self.author: Optional[str] = general_info.get("Created by", {}).get("@value")
        self.references: List[str] = [ref.get("@value") for ref in general_info.get("References to papers", []) if ref.get("@value")]
        self.contact_email: Optional[str] = general_info.get("Contact email", {}).get("@value")

    def _parse_inputs(self) -> List[Dict[str, str]]:
        inputs: List[Dict[str, str]] = []
        for input_feature in self.metadata.get("Input data", []):
            feature = {
                "input_label": input_feature.get("Input label", {}).get("@value", ""),
                "description": input_feature.get("Description", {}).get("@value", ""),
                "type": input_feature.get("Type of input", {}).get("@value", ""),
                "rdfs_label": input_feature.get("Input feature", {}).get("rdfs:label", "")
            }
            inputs.append(feature)
        return inputs

    def _parse_output(self) -> str:
        return self.metadata.get("Outcome label", {}).get("@value", "")

    def __repr__(self) -> str:
        return json.dumps({
            "model_name": self.model_name,
            "description": self.description,
            "docker_image": self.docker_image,
            "inputs": self.inputs,
            "output": self.output,
            "output_label": self.output_label,
            "author": self.author,
            "references": self.references,
            "contact_email": self.contact_email
        }, indent=2)
