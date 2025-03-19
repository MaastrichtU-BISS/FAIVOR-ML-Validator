import json
from typing import List, Dict, Any, Optional


class ModelMetadata:
    def __init__(self, metadata_json: Dict[str, Any]):
        """
        Initialize the ModelMetadata class with provided metadata JSON.

        Parameters
        ----------
        metadata_json : Dict[str, Any]
            A dictionary containing the parsed metadata information.
        """
        self.metadata: Dict[str, Any] = metadata_json
        self.inputs: List[Dict[str, Any]] = self._parse_inputs()
        self.output: str = self._parse_output()
        general_info = self.metadata.get("General Model Information", {})
        self.docker_image: Optional[str] = general_info.get("FAIRmodels image name", {}).get("@value")
        self.model_name: Optional[str] = general_info.get("Title", {}).get("@value")
        self.description: Optional[str] = general_info.get("Editor Note", {}).get("@value")
        self.author: Optional[str] = general_info.get("Created by", {}).get("@value")
        self.references: List[str] = [ref.get("@value") for ref in general_info.get("References to papers", []) if ref.get("@value")]
        self.contact_email: Optional[str] = general_info.get("Contact email", {}).get("@value")

    def _parse_inputs(self) -> List[Dict[str, str]]:
        """Extract input feature details from metadata."""
        inputs: List[Dict[str, str]] = []
        for input_feature in self.metadata.get("Input data", []):
            feature = {
                "description": input_feature.get("Description", {}).get("@value", ""),
                "type": input_feature.get("Type of input", {}).get("@value", ""),
                "rdfs_label": input_feature.get("Input feature", {}).get("rdfs:label", "")
            }
            inputs.append(feature)
        return inputs

    def _parse_output(self) -> str:
        """Extract the output column label from metadata."""
        return self.metadata.get("Outcome", {}).get("rdfs:label", "")

    def __repr__(self) -> str:
        """Provide a JSON-formatted string representation of the metadata."""
        return json.dumps({
            "model_name": self.model_name,
            "description": self.description,
            "docker_image": self.docker_image,
            "inputs": self.inputs,
            "output": self.output,
            "author": self.author,
            "references": self.references,
            "contact_email": self.contact_email
        }, indent=2)
