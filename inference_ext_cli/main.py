"""CLI entry point for inference-ext-cli."""

import click

from inference_ext_cli.generate import generate_command
from inference_ext_cli.profile.command import profile_command
from inference_ext_cli.profile_live import profile_live_command


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CLI tools for RBG inference extension.

    Commands for generating RBG deployment YAMLs with planner integration
    and running SLA profiling pipelines.
    """
    pass


cli.add_command(generate_command)
cli.add_command(profile_command)
cli.add_command(profile_live_command)


if __name__ == "__main__":
    cli()
