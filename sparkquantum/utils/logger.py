import logging

__all__ = ['Logger']


class Logger:
    """Generate a log file and write log data into it."""

    def __init__(self, name, path, level=logging.DEBUG):
        """Build a logger.

        Parameters
        ----------
        name : str
            The name of the class that is providing log data.
        path : str
            The base path for the log file.
        level : enumerate, optional
            The severity level for the log data. Default value is :py:const:`logging.DEBUG`.

        """
        self._name = name
        self._filename = path + 'log.txt'
        self._level = level

    @property
    def name(self):
        """str"""
        return self._name

    @property
    def filename(self):
        """str"""
        return self._filename

    @property
    def level(self):
        """int"""
        return self._level

    def set_level(self, level):
        """Set the severity level for log writes.

        Parameters
        ----------
        level : int
            Severity level for future logs.

        """
        self._level = level

    def _write_message(self, level, name, message):
        with open(self._filename, 'a') as f:
            f.write("{}:{}:{}\n".format(level, name, message))

    def blank(self):
        """Write a blank line in the log file."""
        with open(self._filename, 'a') as f:
            f.write("\n")

    def separator(self):
        """Write a separator line in the log file."""
        with open(self._filename, 'a') as f:
            f.write("# -------------------- #\n")

    def debug(self, message):
        """Write the message in the log file with debug level.

        Parameters
        ----------
        message : str
            Message to be logged.

        """
        if self._level <= logging.DEBUG:
            self._write_message('DEBUG', self._name, message)

    def info(self, message):
        """Write the message in the log file with info level.

        Parameters
        ----------
        message : str
            Message to be logged.

        """
        if self._level <= logging.INFO:
            self._write_message('INFO', self._name, message)

    def warning(self, message):
        """Write the message in the log file with warning level.

        Parameters
        ----------
        message : str
            Message to be logged.

        """
        if self._level <= logging.WARNING:
            self._write_message('WARNING', self._name, message)

    def error(self, message):
        """Write the message in the log file with error level.

        Parameters
        ----------
        message : str
            Message to be logged.

        """
        if self._level <= logging.ERROR:
            self._write_message('ERROR', self._name, message)


def is_logger(obj):
    """Check whether argument is a :py:class:`sparkquantum.utils.logger.Logger` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.utils.logger.Logger` object, False otherwise.

    """
    return isinstance(obj, Logger)
