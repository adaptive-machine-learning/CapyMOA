import click
from jinja2 import Environment, FileSystemLoader
import capymoa  # Needed to initialize JPype with MOA #noqa: F401
from typing import Sequence
from moa.options import AbstractOptionHandler, ClassOption as MoaClassOption
from com.github import javacliparser
import jpype
from dataclasses import dataclass
import re


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclass
class Option:
    name: str
    """Variable name in snake case."""
    doc: str
    """Documentation string for the option."""
    cli_char: str
    """Command line interface character for the option."""
    type: str
    """Python type of the option."""
    default: str
    """Default value of the option as a string."""

    @staticmethod
    def from_javacliparser(option: javacliparser.Option) -> "Option":
        """Convert a ``javacliparser.Option`` to an ``Option`` dataclass."""
        name = camel_to_snake(str(option.getName()))
        doc = str(option.getPurpose())
        cli_char = str(option.getCLIChar())
        type_ = "Any"
        default = "None"

        if isinstance(option, javacliparser.IntOption):
            type_ = "int"
            default = f"{option.getValue()}"
        elif isinstance(option, javacliparser.FloatOption):
            type_ = "float"
            default = f"{option.getValue()}"
        elif isinstance(option, javacliparser.FlagOption):
            type_ = "bool"
            default = "False"
        elif isinstance(option, javacliparser.MultiChoiceOption):
            choices = ", ".join(f'"{choice}"' for choice in option.getOptionLabels())
            type_ = f"Literal[{choices}]"
            default = f'"{option.getChosenLabel()}"'
            definition_list = []
            for label, description in zip(
                option.getOptionLabels(), option.getOptionDescriptions()
            ):
                definition_list.append(f"* ``{label}``: {description}")
            doc += "\n\n" + "\n".join(definition_list)
        elif isinstance(option, MoaClassOption):
            type_ = "str"
            default = f'"{option.getValueAsCLIString()}"'
        else:
            raise NotImplementedError(f"Option type {type(option)} not implemented")

        return Option(
            name=name, cli_char=cli_char, doc=doc, type=type_, default=default
        )


def get_options(abstract_options: AbstractOptionHandler) -> Sequence[Option]:
    """Get the options of an object as a list."""
    options = abstract_options.getOptions().getOptionArray()
    return [Option.from_javacliparser(opt) for opt in options]


@click.command()
@click.argument("java_classifier", type=str, required=True)
def main(java_classifier: str):
    environment = Environment(loader=FileSystemLoader("."))
    environment.filters["camel_to_snake"] = camel_to_snake

    # Construct the Java object
    j_object = jpype.JClass(java_classifier)()

    # Render the template to stdout
    template = environment.get_template("MOAClassifier.py.jinja")
    print(
        template.render(
            options=get_options(j_object),
            j_class=java_classifier.split(".")[-1],
            j_package=java_classifier.rsplit(".", 1)[0],
        )
    )


if __name__ == "__main__":
    main()
