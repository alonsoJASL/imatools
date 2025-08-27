import subprocess
import logging
import os
import sys

from imatools.common.config import configure_logging, add_file_handler

class CommandRunner:
    def __init__(self, debug=False, log_dir='', verbose=True):
        """
        Initialize the CommandRunner class.

        Args:
            debug (bool): If True, print command outputs to console.
            log_dir (str): Directory for log files.
        """
        self._debug = debug
        self._log_dir = log_dir
        self._verbose = verbose

        # Configure the logger
        self.logger = configure_logging("CommandRunner", log_level=logging.INFO)

    # setters and getters
    def set_debug(self, debug):
        self._debug = debug
    
    def get_debug(self):
        return self._debug
    
    def verbose(self):
        return self._verbose
    
    def set_log_dir(self, log_dir):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        error_log_path = os.path.join(log_dir, "errors.log")
        add_file_handler(self.logger, error_log_path, log_level=logging.ERROR)

    def run_command(self, command, expected_outputs=None):
        """
        Run a command-line command and handle errors.

        Args:
            command (str): The command to execute.
            expected_outputs (list[str]): List of expected file paths or outputs.

        Raises:
            Exception: If the command fails or expected outputs are missing.

        Returns:
            str: The stdout of the command.
        """
        command_name = command.split(' ')[0].split('/')[-1]
        if self._verbose:
            self.logger.info(f"Running command:\t{command_name}")
            self.logger.info(command)
        
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if self._debug:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            # Check if the known benign error appears in stderr.
            if "Caught exception 'unordered_map::at()'" in result.stderr:
                self.logger.warning(
                    f"Ignoring known non-critical error in command '{command_name}': {result.stderr.strip()}"
                )
            else :
                err_msg = f"Error running command: {command}"
                if os.path.exists(self._log_dir):
                    error_log_name = os.path.join(self._log_dir, f"{command_name}_error.log")
                    self._write_error_log(error_log_name, command, result)
                    err_msg += f" (see {error_log_name})"
                self.logger.error(f"Error running command: {command}")
                raise Exception(err_msg)
            
        else:
            if self._verbose:
                self.logger.info(f"{command_name} successful")

        if expected_outputs:
            self._validate_outputs(command_name, expected_outputs)

        return result.stdout
    
    def build_command(self, command, arguments:list) -> str:
        """
        Build a command string with arguments.

        Args:
            command (str): The command to execute.
            arguments (list[str]): List of arguments to add to the command.

        Returns:
            str: The command string.
        """
        return f"{command} {' '.join(arguments)}"

    def _write_error_log(self, log_path, command, result):
        """
        Write an error log for a failed command.

        Args:
            log_path (str): Path to the log file.
            command (str): The command that was executed.
            result (subprocess.CompletedProcess): The result of the subprocess.run call.
        """
        with open(log_path, "w") as f:
            f.write(f"Error running command: {command}\n\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write(f"stdout:\n{result.stdout}\n\n")
            f.write(f"stderr:\n{result.stderr}\n\n")
        self.logger.info(f"Error log written to {log_path}")

    def _validate_outputs(self, command_name, expected_outputs):
        """
        Validate that expected outputs are generated.

        Args:
            command_name (str): Name of the command that was run.
            expected_outputs (list[str]): List of expected file paths.

        Raises:
            Exception: If one or more expected outputs are missing.
        """
        missing_outputs = [output for output in expected_outputs if not os.path.exists(output)]
        if missing_outputs:
            error_log_name = os.path.join(self._log_dir, f"{command_name}_missing_outputs.log")
            self.logger.error(f"Missing expected outputs from {command_name}: {missing_outputs}")
            with open(error_log_name, "w") as f:
                f.write(f"Command: {command_name}\n")
                f.write("Missing outputs:\n")
                f.writelines(f"{output}\n" for output in missing_outputs)

            raise Exception(f"Missing outputs: {missing_outputs} (see {error_log_name})")
       
