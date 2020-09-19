import csv
import json
from datetime import datetime
from urllib import error, request

from pyspark import SparkContext

from sparkquantum import conf, util

__all__ = ['Profiler', 'is_profiler']


class Profiler:
    """Profile and export the resources consumed by Spark_.

    .. _Spark:
        https://spark.apache.org

    """

    def __init__(self):
        """Build a profiler object."""
        self._sc = SparkContext.getOrCreate()

        self._data = None
        self._resources = None
        self._executors = None

        self._base_url = self._get_baseurl()
        self._enabled = self._is_enabled()

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)

        self._start()

    @staticmethod
    def _default_rdd():
        return {'memoryUsed': 0, 'diskUsed': 0}

    @staticmethod
    def _default_resources():
        return {'totalShuffleWrite': [], 'totalShuffleRead': [],
                'diskUsed': [], 'memoryUsed': []}

    @staticmethod
    def _default_executor():
        return {'totalShuffleWrite': [], 'totalShuffleRead': [],
                'diskUsed': [], 'memoryUsed': []}

    @staticmethod
    def _export_values(values, fieldnames, filename, extension='csv'):
        if extension == 'csv':
            f = filename + '.' + extension

            with open(f, 'w') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

                for v in values:
                    w.writerow(v)
        else:
            raise NotImplementedError("unsupported file extension")

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        """Build a string representing this profiler.

        Returns
        -------
        str
            The string representation of this profiler.

        """
        return 'Profiler configured to request data from {}'.format(
            self._base_url)

    def _is_enabled(self):
        return conf.get(self._sc,
                        'sparkquantum.profiling.enabled') == 'True'

    def _get_baseurl(self):
        return conf.get(self._sc,
                        'sparkquantum.profiling.baseUrl')

    def _start(self):
        self._data = {}
        self._resources = self._default_resources()
        self._executors = {}

    def reset(self):
        """Reset the profiler attributes to get info for a new profiling round."""
        self._start()

    def _request(self, url_suffix=''):
        url = self._base_url + 'applications' + url_suffix

        self._logger.info("performing request to '{}'...".format(url))

        t1 = datetime.now()

        try:
            with request.urlopen(url) as response:
                result = response.read()

            self._logger.debug("request performed in {}s".format(
                (datetime.now() - t1).total_seconds()))
        except error.URLError as e:
            self._logger.warning(
                "request to {} failed with the following error: '{}' and no data will be returned".format(url, e.reason))
            return None

        if result is not None:
            result = json.loads(result.decode('utf-8'))
        else:
            self._logger.warning(
                "the response of request to {} is empty".format(url))
        return result

    def request_applications(self):
        """Request application info.

        Returns
        -------
        list
            A list with application info if profiling is enabled, `None` otherwise.

        """
        if self._enabled:
            return self._request()

    def request_jobs(self, app_id, job_id=None):
        """Request an application's jobs info.

        Parameters
        ----------
        app_id : int
            The application's id.
        job_id : int, optional
            The job's id.

        Returns
        -------
        list or dict
            A list with all application's jobs info or a dict with the job info if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            if job_id is None:
                return self._request("/{}/jobs".format(app_id))
            else:
                return self._request("/{}/jobs/{}".format(app_id, job_id))

    def request_stages(self, app_id, stage_id=None):
        """Request an application's stages info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int, optional
            The stage's id.

        Returns
        -------
        list or dict
            A list with all application's stages info or a dict with the stage info if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            if stage_id is None:
                return self._request("/{}/stages".format(app_id))
            else:
                return self._request("/{}/stages/{}".format(app_id, stage_id))

    def request_stageattempt(self, app_id, stage_id, stageattempt_id):
        """Request an application's stage attempts info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int
            The stage's id.
        stageattempt_id : int
            The stage attempt's id.

        Returns
        -------
        dict
            A dict with an application's stage attempt info if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            return self._request(
                "/{}/stages/{}/{}".format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasksummary(
            self, app_id, stage_id, stageattempt_id):
        """Request the task summary of a stage attempt info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int
            The stage's id.
        stageattempt_id : int
            The stage attempt's id.

        Returns
        -------
        dict
            A dict with the task summary of a stage attempt if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            return self._request(
                "/{}/stages/{}/{}/taskSummary".format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasklist(self, app_id, stage_id, stageattempt_id):
        """Request the task list of a stage attempt info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int
            The stage's id.
        stageattempt_id : int
            The stage attempt's id.

        Returns
        -------
        list
            A list with the task list of a stage attempt if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            return self._request(
                "/{}/stages/{}/{}/taskList".format(app_id, stage_id, stageattempt_id))

    def request_executors(self, app_id):
        """Request an application's active executors info.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        list
            A list with all application's active executors info if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            return self._request("/{}/executors".format(app_id))

    def request_allexecutors(self, app_id):
        """Request an application's executors info.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        list
            A list with all application's executors info if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            return self._request("/{}/allexecutors".format(app_id))

    def request_rdd(self, app_id, rdd_id=None):
        """Request an application's RDD info.

        Parameters
        ----------
        app_id : int
            The application's id.
        rdd_id : int, optional
            The RDD's id.

        Returns
        -------
        list or dict
            A list with all application's RDD info or a dict with the RDD info if profiling is enabled,
            `None` otherwise or when an error occurred.

        """
        if self._enabled:
            if rdd_id is None:
                return self._request("/{}/storage/rdd".format(app_id))
            else:
                return self._request(
                    "/{}/storage/rdd/{}".format(app_id, rdd_id))

    def log_executors(self, data=None, app_id=None):
        """Log all executors info into the log file.

        Notes
        -----
        When no data is provided, the application's id is used to request those data.

        Parameters
        ----------
        data : list, optional
            The executors data.
        app_id : int
            The application's id.

        Raises
        ------
        ValueError
            If `app_id` is not valid.

        """
        if data is None:
            if app_id is None:
                self._logger.error(
                    "application id expected, not '{}'".format(type(app_id)))
                raise ValueError(
                    "application id expected, not '{}'".format(type(app_id)))
            data = self.request_executors(app_id)

        if data is not None:
            self._logger.info("printing executors data...")
            for d in data:
                for k, v in d.items():
                    self._logger.info("{}: {}".format(k, v))

    def log_rdd(self, data=None, app_id=None, rdd_id=None):
        """Log all RDD info into the log file.

        Notes
        -----
        When no data is provided, the application's id is used to request all its RDD data.
        If the RDD's id are also provided, they are used to get its data.

        Parameters
        ----------
        data : list, optional
            The executors data.
        app_id : int, optional
            The application's id.
        rdd_id : int, optional
            The RDD's id.

        Raises
        ------
        ValueError
            If `app_id` is not valid.

        """
        if data is None:
            if app_id is None:
                self._logger.error(
                    "expected an application id, not '{}'".format(type(app_id)))
                raise ValueError(
                    "expected an application id, not '{}'".format(type(app_id)))
            else:
                if rdd_id is None:
                    data = self.request_rdd(app_id)

                    if data is not None:
                        self._logger.info("printing RDDs data...")
                        for d in data:
                            for k, v in d.items():
                                if k != 'partitions':
                                    self._logger.info(
                                        "{}: {}".format(k, v))
                else:
                    data = self.request_rdd(app_id, rdd_id)

                    if data is not None:
                        self._logger.info(
                            "printing RDD (id {}) data...".format(rdd_id))
                        for k, v in data.items():
                            if k != 'partitions':
                                self._logger.info("{}: {}".format(k, v))

    def profile_rdd(self, name, app_id, rdd_id):
        """Store information about a RDD that represents a quantum walk element.

        Parameters
        ----------
        name : str
            A name for the element.
        app_id : int
            The application's id.
        rdd_id : int
            The RDD's id.

        """
        if self._enabled:
            self._logger.info("profiling RDD for '{}'...".format(name))

            if name not in self._data:
                self._data[name] = self._default_rdd()

            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in self._data[name]:
                        self._data[name][k] = v

    def profile_resources(self, app_id):
        """Store information about the resources consumed by the application.

        Parameters
        ----------
        app_id : int
            The application's id.

        """
        if self._enabled:
            self._logger.info("profiling application resources...")

            data = self.request_executors(app_id)

            for k in self._resources:
                self._resources[k].append(0)

            if data is not None:
                for d in data:
                    for k, v in d.items():
                        if k in self._resources:
                            self._resources[k][-1] += v

    def profile_executors(self, app_id, exec_id=None):
        """Store all executors info.

        Notes
        -----
        When no executor's id is provided, all the application's executors info are requested and stored.

        Parameters
        ----------
        app_id : int
            The application's id.
        exec_id : int, optional
            The executor's id.

        """
        if self._enabled:
            if exec_id is None:
                self._logger.info("profiling resources of executors...")
            else:
                self._logger.info(
                    "profiling resources of executor {}...".format(exec_id))

            data = self.request_executors(app_id)

            if data is not None:
                if exec_id is None:
                    for d in data:
                        if d['id'] not in self._executors:
                            self._executors[d['id']] = self._default_executor()

                        for k, v in d.items():
                            if k in self._executors[d['id']]:
                                self._executors[d['id']][k].append(v)
                else:
                    for d in data:
                        if d['id'] == exec_id:
                            if d['id'] not in self._executors:
                                self._executors[d['id']
                                                ] = self._default_executor()

                            for k, v in d.items():
                                if k in self._executors[d['id']]:
                                    self._executors[d['id']][k].append(v)
                            break

    def get_rdd(self, name=None):
        """Get the RDD resources of all elements or of the named one.

        Parameters
        ----------
        name : str, optional
            A name for the element.

        Returns
        -------
        dict
            A dict with the RDD resources of all elements or with just the RDD resources of the named element.

        """
        if len(self._data):
            if name is None:
                return self._data.copy()
            else:
                if name not in self._data:
                    self._logger.warning(
                        "no measurement of RDD resources has been done for '{}'".format(name))
                    return {}
                return self._data[name]
        else:
            self._logger.warning(
                "no measurement of RDD resources has been done")
            return {}

    def get_resources(self, name=None):
        """Get the resources of all elements or of the named one.

        Parameters
        ----------
        name : str, optional
            The name representing a resource.

        Returns
        -------
        dict
            A dict with the resources of all elements or with just the resources of the named element.

        """
        if len(self._resources):
            if name is None:
                return self._resources.copy()
            else:
                if name not in self._default_resources():
                    self._logger.warning(
                        "no measurement of resources has been done for '{}'".format(name))
                    return {}
                return self._resources[name]
        else:
            self._logger.warning(
                "no measurement of resources has been done")
            return {}

    def get_executors(self, exec_id=None):
        """Get the resources of all executors or of the named one (by its id).

        Parameters
        ----------
        exec_id : int, optional
            An executor id.

        Returns
        -------
        dict
            A dict with the resources of all executors or with just the resources of the named executor.

        """
        if len(self._executors):
            if exec_id is None:
                return self._executors.copy()
            else:
                if exec_id not in self._executors:
                    self._logger.warning(
                        "no measurement of resources has been done for executor {}".format(exec_id))
                    return {}
                return self._executors[exec_id]
        else:
            self._logger.warning(
                "no measurement of executors resources has been done")
            return self._executors

    def export_rdd(self, path, extension='csv'):
        """Export all stored RDD resources informations.

        Notes
        -----
        For now, only CSV extension is supported.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv'.

        Raises
        ------
        NotImplementedError
            If `extension` is not valid or not supported.

        """
        self._logger.info(
            "exporting RDD resources in {} format...".format(extension))

        if len(self._data):
            rdd = []

            for k, v in self._data.items():
                tmp = v.copy()
                tmp['rdd'] = k
                rdd.append(tmp)

            self._export_values(rdd, rdd[-1].keys(),
                                util.append_slash(path) + 'rdd', extension)

            self._logger.info("RDD resources successfully exported")
        else:
            self._logger.warning(
                "no measurement of RDD resources has been done")

    def export_resources(self, path, extension='csv'):
        """Export all stored resources informations.

        Notes
        -----
        For now, only CSV extension is supported.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv'.

        Raises
        ------
        NotImplementedError
            If `extension` is not valid or not supported.

        """
        self._logger.info(
            "exporting resources in {} format...".format(extension))

        if len(self._resources):
            resources = []
            size = 0

            for k, v in self._resources.items():
                size = len(v)
                break

            for i in range(size):
                tmp = self._default_resources()

                for k in tmp:
                    tmp[k] = self._resources[k][i]

                resources.append(tmp)

            self._export_values(resources, resources[-1].keys(),
                                util.append_slash(path) + 'resources', extension)

            self._logger.info("resources successfully exported")
        else:
            self._logger.warning(
                "no measurement of resources has been done")

    def export_executors(self, path, extension='csv'):
        """Export all stored executors' resources informations.

        Notes
        -----
        For now, only CSV extension is supported.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv'.

        Raises
        ------
        NotImplementedError
            If `extension` is not valid or not supported.

        """
        self._logger.info(
            "exporting executors resources in {} format...".format(extension))

        if len(self._executors):
            for k1, v1 in self._executors.items():
                executors = []
                size = 0

                for k2, v2 in v1.items():
                    size = len(v2)
                    break

                for i in range(size):
                    tmp = self._default_resources()

                    for k2 in tmp:
                        tmp[k2] = v1[k2][i]

                    executors.append(tmp)

                self._export_values(executors, executors[-1].keys(),
                                    "{}executor_{}".format(util.append_slash(path), k1), extension)

            self._logger.info("executors resources successfully exported")
        else:
            self._logger.warning(
                "no measurement of executors resources has been done")

    def export(self, path, extension='csv'):
        """Export all stored profiling information.

        Notes
        -----
        For now, only CSV extension is supported.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv'.

        Raises
        ------
        NotImplementedError
            If `extension` is not valid or not supported.

        """
        self.export_rdd(path, extension)
        self.export_resources(path, extension)
        self.export_executors(path, extension)


def is_profiler(obj):
    """Check whether argument is a :py:class:`sparkquantum.profiler.Profiler` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.profiler.Profiler` object, False otherwise.

    """
    return isinstance(obj, Profiler)
