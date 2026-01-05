"""Helper function for loading personas based on yaml descriptions."""
import argparse
from enum import Enum
import os
import pydantic
from typing import Union
from pathlib import Path
import yaml
import logging
from collections import defaultdict

class Persona(pydantic.BaseModel):
    name: str
    age: int
    gender: Gender
    occupation: str
    # Free-form description.
    description: str
    # List of short description of the current focus of the persona.
    focus: list[str]
    attrs: set[str]

    def __hash__(self):
        return hash(self.description)

    def get_summary(self) -> str:
        """To provide enough context to answer a survey."""
        return f"""
        <begin desc>
        age: {self.age}
        gender: {self.gender}
        occupation: {self.occupation}
        description: {self.description}
        focus: {self.focus}
        <end desc>
        """

def load_personas(path: Union[str, Path]) -> list[Persona]:
    personas = []
    p = Path(path)
    if not p.is_dir():
        logging.error(f"Error: Path '{path}' is not a valid directory.")
        return personas

    for file_path in p.glob('*.yaml'):
        logging.info(f"Attempting to load personas from: {file_path.name}")
        group_name = file_path.stem
        personas_in_file = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            for persona in data:
                persona = Persona.model_validate(persona)
                personas_in_file.append(persona)
                logging.info(f"Successfully loaded: '{persona.name}'")

        except pydantic.ValidationError as e:
            logging.warning(f"Skipping file: '{file_path.name}' (Schema validation failed)\n{e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred with file '{file_path.name}': {e}")
        
        personas.extend(personas_in_file)
        
    return personas


def main():
    parser = argparse.ArgumentParser(description="Load and display personas")
    parser.add_argument(
        "--path",
        type=str,
        default="data/happiness_unhappiness/personas",
        help="personas path"
    )
    args = parser.parse_args()
    personas = load_personas(args.path)
    print(f"#################### Personas ####################")
    for persona in personas:
        print(persona)

if __name__ == "__main__":
    main()