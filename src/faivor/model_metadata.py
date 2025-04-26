from dataclasses import dataclass
import json
from typing import List, Dict, Any, Optional

@dataclass
class ModelInput:
    """
    Class to represent a model input feature.
    """
    input_label: str
    description: str
    data_type: str
    rdfs_label: str
    def __post_init__(self):
        """
        Validate the input feature attributes are strings or None.
        """
        if not isinstance(self.input_label, str):
            raise TypeError("input_label must be a string")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if self.data_type is not None and not isinstance(self.data_type, str):
            raise TypeError("type must be a string")
        if self.rdfs_label is not None and  not isinstance(self.rdfs_label, str):
            raise TypeError("rdfs_label must be a string")

class ModelMetadata:
    """
    Class to store metadata information for a ML model, following FAIRmodels syntax.
    """

    def __init__(self, metadata_json: Dict[str, Any]):
        if not isinstance(metadata_json, dict):
            raise TypeError("Expected a dictionary for metadata_json")            
        self.metadata = metadata_json
        if "General Model Information" not in metadata_json:
            raise ValueError("Missing required 'General Model Information' section in metadata")
    
        self.inputs: List[ModelInput] = self._parse_inputs()
        self.output: str = self._parse_output()
        self.output_label: Optional[str] = metadata_json.get("Outcome label", {}).get("@value")

        general_info = self.metadata.get("General Model Information", {})
        self.docker_image: Optional[str] = general_info.get("FAIRmodels image name", {}).get("@value")
        self.model_name: Optional[str] = general_info.get("Title", {}).get("@value")
        self.description: Optional[str] = general_info.get("Editor Note", {}).get("@value")
        self.author: Optional[str] = general_info.get("Created by", {}).get("@value")
        self.references: List[str] = [ref.get("@value") for ref in general_info.get("References to papers", []) if ref.get("@value")]
        self.contact_email: Optional[str] = general_info.get("Contact email", {}).get("@value")

    def validate(self) -> bool:
        """
        Validate the metadata making sure it contains all required fields.
        
        Returns:
            bool: True if metadata is valid, False otherwise
        """
        required_fields = ["model_name", "inputs", "output"]
        return all(getattr(self, field) for field in required_fields)
    
    def _parse_inputs(self) -> List[ModelInput]:
        """
        Extract and normalize input features from the metadata.
        
        Returns:
            List of ModelInput objects containing normalized input feature information.
        """
        
        inputs: List[ModelInput] = []
        for input_feature in self.metadata.get("Input data", []):
            feature = ModelInput(
                input_label = input_feature.get("Input label", {}).get("@value", ""),
                description =  input_feature.get("Description", {}).get("@value", ""),
                data_type = input_feature.get("Type of input", {}).get("@value", ""),
                rdfs_label = input_feature.get("Input feature", {}).get("rdfs:label", "")
            )
            inputs.append(feature)
        return inputs

    def _parse_output(self) -> str:
        """
        Extract the model's output label from the metadata.
        
        Returns:
            str: The output label value from the 'Outcome label' field
        """
                
        return self.metadata.get("Outcome label", {}).get("@value", "")

    def __repr__(self) -> str:
        """
        Create a string representation of the class object.
        
        Returns a JSON string containing the key attributes of the
        model metadata
        
        Returns:
            str: JSON string representation of the model metadata
        """
                
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

