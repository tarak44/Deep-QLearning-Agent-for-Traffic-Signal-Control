from pathlib import Path
from typing import Annotated

import typer

from tlcs.constants import (
    DEFAULT_EVAL_FOLDER,
    DEFAULT_MODEL_PATH,
    DEFAULT_SETTINGS_PATH,
    DEFAULT_TEST_FOLDER,
    TESTING_SETTINGS_FILE,
    TRAINING_SETTINGS_FILE,
)
from tlcs.main import (
    baseline_session,
    evaluation_session,
    testing_session,
    training_session,
)

app = typer.Typer(
    help="Train and run TLCS.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _confirm_overwrite_directory(
    directory: Path,
    overwrite_message: str,
    cancel_message: str,
) -> None:
    """Ask the user to confirm overwriting an existing directory.

    Args:
        directory: The directory that potentially will be overwritten.
        overwrite_message: The message prompting the user to confirm overwrite.
        cancel_message: The message displayed if the user cancels.
    """
    typer.echo(f"⚠️  The folder '{directory}' already exists.")
    confirm = typer.confirm(overwrite_message, default=False)
    if not confirm:
        typer.echo(cancel_message)
        raise typer.Abort


def check_training_path(out_path: Path) -> None:
    """Ensure training output path is safe to use.

    Args:
        out_path: Directory where training outputs and model will be saved.
    """
    if out_path.exists():
        _confirm_overwrite_directory(
            directory=out_path,
            overwrite_message=(
                "Continuing will possibly overwrite the existing training data and model. "
                "Do you want to continue?"
            ),
            cancel_message="Training cancelled.",
        )


def check_testing_path(model_path: Path, test_name: str) -> None:
    """Ensure testing model path and test folder are valid.

    Args:
        model_path: Path to the directory containing the trained model.
        test_name: Name of the test folder to create within the model directory.
    """
    model_files = list(model_path.glob("*.pt"))

    if not model_files:
        typer.echo(f"Model file (*.pt) not found in model path: '{model_path}'.")
        raise typer.Abort

    test_folder = model_path / test_name

    if test_folder.exists():
        _confirm_overwrite_directory(
            directory=test_folder,
            overwrite_message=(
                "Continuing will overwrite the content of the test folder. Do you want to continue?"
            ),
            cancel_message="Testing cancelled.",
        )


def check_eval_path(model_path: Path, eval_name: str) -> None:
    """Ensure evaluation model path and eval folder are valid.

    Args:
        model_path: Path to the directory containing the trained model.
        eval_name: Name of the evaluation folder to create within the model directory.
    """
    model_files = list(model_path.glob("*.pt"))

    if not model_files:
        typer.echo(f"Model file (*.pt) not found in model path: '{model_path}'.")
        raise typer.Abort

    eval_folder = model_path / eval_name

    if eval_folder.exists():
        _confirm_overwrite_directory(
            directory=eval_folder,
            overwrite_message=(
                "Continuing will overwrite the content of the evaluation folder. "
                "Do you want to continue?"
            ),
            cancel_message="Evaluation cancelled.",
        )


@app.command(
    name="train", help="Train a new TLCS model using the specified settings file."
)
def cmd_train(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing training parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH
    / TRAINING_SETTINGS_FILE,
    out_path: Annotated[
        Path,
        typer.Option(
            help="Directory where training outputs and trained model will be saved.",
        ),
    ] = DEFAULT_MODEL_PATH,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Optional global seed to make training reproducible.",
        ),
    ] = None,
    resume_from: Annotated[
        Path | None,
        typer.Option(
            help="Optional checkpoint path to resume training from.",
        ),
    ] = None,
) -> None:
    """CLI command to train a TLCS model.

    Args:
        settings_file: Path to the YAML file with training parameters.
        out_path: Output directory for training artifacts and the trained model.
        seed: Optional global seed value to override the settings file.
        resume_from: Optional checkpoint path for resuming training.
    """
    if resume_from is None:
        check_training_path(out_path)
    elif not resume_from.exists():
        typer.echo(f"Checkpoint file not found: '{resume_from}'.")
        raise typer.Abort

    training_session(
        settings_file=settings_file,
        out_path=out_path,
        seed_override=seed,
        resume_from=resume_from,
    )


@app.command(name="test", help="Run a simulation test using a trained TLCS model.")
def cmd_test(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH
    / TESTING_SETTINGS_FILE,
    model_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the directory containing the trained TLCS model to test.",
        ),
    ] = DEFAULT_MODEL_PATH,
    test_name: Annotated[
        str,
        typer.Option(
            help="The name of the test folder.",
        ),
    ] = DEFAULT_TEST_FOLDER,
) -> None:
    """CLI command to run a simulation test using a trained TLCS model.

    Args:
        settings_file: Path to the YAML file with testing parameters.
        model_path: Path to the directory containing the trained model.
        test_name: Name of the test folder created under the model directory.
    """
    check_testing_path(model_path=model_path, test_name=test_name)
    testing_session(
        settings_file=settings_file, model_path=model_path, test_name=test_name
    )


@app.command(name="eval", help="Evaluate a trained TLCS model across multiple seeds.")
def cmd_eval(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH
    / TESTING_SETTINGS_FILE,
    model_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the directory containing the trained TLCS model to evaluate.",
        ),
    ] = DEFAULT_MODEL_PATH,
    eval_name: Annotated[
        str,
        typer.Option(
            help="The name of the evaluation folder.",
        ),
    ] = DEFAULT_EVAL_FOLDER,
    seeds: Annotated[
        list[int] | None,
        typer.Option(
            help="Optional list of seeds, repeat the option to pass multiple seeds.",
        ),
    ] = None,
) -> None:
    """CLI command to evaluate a TLCS model across multiple seeds."""
    check_eval_path(model_path=model_path, eval_name=eval_name)
    evaluation_session(
        settings_file=settings_file,
        model_path=model_path,
        eval_name=eval_name,
        seeds=seeds,
    )


@app.command(
    name="baseline", help="Evaluate a fixed-time baseline across multiple seeds."
)
def cmd_baseline(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH
    / TESTING_SETTINGS_FILE,
    eval_name: Annotated[
        str,
        typer.Option(
            help="The name of the baseline output folder (under model/).",
        ),
    ] = "baseline",
    seeds: Annotated[
        list[int] | None,
        typer.Option(
            help="Optional list of seeds, repeat the option to pass multiple seeds.",
        ),
    ] = None,
) -> None:
    """CLI command to evaluate the fixed-time baseline."""
    baseline_session(settings_file=settings_file, eval_name=eval_name, seeds=seeds)


if __name__ == "__main__":
    app()
