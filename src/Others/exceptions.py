class CustomException(Exception):
    """
    A custom exception class that formats error messages with color and styling.

    Attributes:
        message (str): The error message.
        exception_type (str): The type of the exception. Defaults to "Error".
    """

    def __init__(self, message: str, exception_type: str = "Error"):
        color_code = "\033[91m"  # Red color
        reset_code = "\033[0m"  # Reset color
        formatted_message = (
            f"\n{'=' * 30}\n"
            f"❗️❗️❗️ [{exception_type}] {message} ❗️❗️❗️\n"
            f"{'=' * 30}\n"
        )
        super().__init__(f"{color_code}{formatted_message}{reset_code}")

class ArgumentException(CustomException):
    def __init__(self, message):
        super().__init__(message, exception_type="Argument Exception")

class DataException(CustomException):
    def __init__(self, message):
        super().__init__(message, exception_type="Data Exception")


class ModelException(CustomException):
    def __init__(self, message):
        super().__init__(message, exception_type="Model Exception")


class MetricException(CustomException):
    def __init__(self, message):
        super().__init__(message, exception_type="Metric Exception")


class TrainingException(CustomException):
    def __init__(self, message):
        super().__init__(message, exception_type="Train Exception")


class EnviromentException(CustomException):
    def __init__(self, message):
        super().__init__(message, exception_type="Enviroment Exception")
