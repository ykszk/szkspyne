import subprocess
import shlex
from typing import List, Optional, Union, Callable


class Pipeline:
    """Class to represent a pipeline of shell commands."""

    def __init__(self):
        """Initialize an empty pipeline."""
        self.commands: List[List[str]] = []

    def add_command(self, command: Union[str, List[str]]) -> 'Pipeline':
        """
        Add a command to the pipeline.

        Args:
            command: Command as a string or list of arguments

        Returns:
            Self for method chaining
        """
        if isinstance(command, str):
            self.commands.append(shlex.split(command))
        else:
            self.commands.append(command)
        return self

    def pipe(self, command: Union[str, List[str]]) -> 'Pipeline':
        """Alias for add_command for a more fluid API."""
        return self.add_command(command)

    def run(self, input_data: Optional[str] = None, text: bool = True) -> str:
        """
        Execute the pipeline.

        Args:
            input_data: Optional input data for the first command
            text: Whether to return the output as text (True) or bytes (False)

        Returns:
            Output of the last command in the pipeline
        """
        if not self.commands:
            return ""

        # Convert input to bytes if provided and not already bytes
        stdin_data = None
        if input_data is not None:
            if isinstance(input_data, str) and not text:
                stdin_data = input_data.encode()
            elif isinstance(input_data, str) and text:
                stdin_data = input_data.encode()
            else:
                stdin_data = input_data

        # For the first command
        if len(self.commands) == 1:
            result = subprocess.run(self.commands[0], input=stdin_data, stdout=subprocess.PIPE, text=text)
            return result.stdout

        # For multiple commands
        previous_process = None

        for i, cmd in enumerate(self.commands):
            is_first = i == 0
            is_last = i == len(self.commands) - 1

            # Set up the process
            if is_first:
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=text)
                if stdin_data:
                    process.stdin.write(stdin_data)  # type: ignore
                process.stdin.close()  # type: ignore
            elif is_last:
                process = subprocess.Popen(cmd, stdin=previous_process.stdout, stdout=subprocess.PIPE, text=text)  # type: ignore
                previous_process.stdout.close()  # type: ignore
            else:
                process = subprocess.Popen(cmd, stdin=previous_process.stdout, stdout=subprocess.PIPE, text=text)  # type: ignore
                previous_process.stdout.close()  # type: ignore

            previous_process = process

        # Get output from the last command
        output, _ = previous_process.communicate()  # type: ignore
        return output

    def run_to_file(self, filename: str, input_data: Optional[str] = None, text: bool = True) -> None:
        """
        Execute the pipeline and save the output to a file.

        Args:
            filename: Output filename
            input_data: Optional input data for the first command
            text: Whether to return the output as text (True) or bytes (False)
        """
        output = self.run(input_data, text)
        with open(filename, 'w') as f:
            f.write(output)

    def runner(self, input_data: Optional[str] = None, text: bool = True, output_file=None) -> Callable:
        """
        Return a callable that runs the pipeline.
        This is useful for use with doit.

        Args:
            input_data: Optional input data for the first command
            text: Whether to return the output as text (True) or bytes (False)
        """
        # return lambda: self.run(input_data, text)
        if output_file:
            return lambda: self.run_to_file(output_file, input_data, text)
        else:
            return lambda: self.run(input_data, text)
