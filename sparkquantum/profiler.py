import csv
import json
from datetime import datetime
from urllib import error, request

from pyspark import SparkContext

from sparkquantum import conf, util

__all__ = ['Profiler', 'is_profiler']


class Profiler:
    """Top-level class for Spark_ metrics profilers.

    .. _Spark:
        https://spark.apache.org

    """

    def __init__(self):
        """Build a top-level profiler object."""
        self._sc = SparkContext.getOrCreate()

        self._rdd = None
        self._executors = None

        self._base_url = self._get_baseurl()
        self._enabled = self._is_enabled()

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)

        self._start()

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

    def _rdd_metrics(self):
        return {'numPartitions': 0,
                'numCachedPartitions': 0,
                'memoryUsed': 0,
                'diskUsed': 0}

    def _executor_metrics(self):
        return {'memoryUsed': 0,
                'diskUsed': 0,
                'maxMemory': 0,
                'totalShuffleWrite': 0,
                'totalShuffleRead': 0,
                'memoryMetrics.usedOnHeapStorageMemory': 0,
                'memoryMetrics.usedOffHeapStorageMemory': 0,
                'memoryMetrics.totalOnHeapStorageMemory': 0,
                'memoryMetrics.totalOffHeapStorageMemory': 0,
                'peakMemoryMetrics.OnHeapExecutionMemory': 0,
                'peakMemoryMetrics.OffHeapExecutionMemory': 0,
                'peakMemoryMetrics.OnHeapStorageMemory': 0,
                'peakMemoryMetrics.OffHeapStorageMemory': 0,
                'peakMemoryMetrics.OnHeapUnifiedMemory': 0,
                'peakMemoryMetrics.OffHeapUnifiedMemory': 0}

    def _get_baseurl(self):
        return conf.get(self._sc, 'sparkquantum.profiling.baseUrl')

    def _is_enabled(self):
        return util.to_bool(
            conf.get(self._sc, 'sparkquantum.profiling.enabled'))

    def _start(self):
        self._rdd = {}
        self._executors = {}

    def _export(self, values, fieldnames, filename, extension='csv'):
        if extension == 'csv':
            f = filename + '.' + extension

            with open(f, 'w') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

                for v in values:
                    w.writerow(v)
        else:
            raise NotImplementedError("unsupported file extension")

    def _request(self, url_suffix=''):
        url = self._base_url + 'applications' + url_suffix

        self._logger.info("performing request to '{}'...".format(url))

        time = datetime.now()

        try:
            with request.urlopen(url) as response:
                result = response.read()

            time = (datetime.now() - time).total_seconds()

            self._logger.debug("request performed in {}s".format(time))
        except error.URLError as e:
            self._logger.warning(
                "request to {} failed with the following error: '{}' and no data will be returned".format(url, e.reason))
            return None

        if result is not None:
            result = json.loads(result.decode('utf-8'))
        else:
            self._logger.warning(
                "the response of request to {} was empty".format(url))
        return result

    def request_applications(self):
        """Request all applications' data.

        Returns
        -------
        list
            A list with all applications' data if profiling is enabled, `None` otherwise.

        """
        if self._enabled:
            return self._request()

    def request_jobs(self, app_id, job_id=None):
        """Request all the application's jobs' data.

        Parameters
        ----------
        app_id : int
            The application's id.
        job_id : int, optional
            The job's id. Default value is None.

        Returns
        -------
        list or dict
            A list with all the application's jobs' data or a dict with a
            job's data if profiling is enabled, `None` otherwise.

        """
        if self._enabled:
            if job_id is None:
                return self._request("/{}/jobs".format(app_id))
            else:
                return self._request("/{}/jobs/{}".format(app_id, job_id))

    def request_stages(self, app_id, stage_id=None):
        """Request all the application's stages' data.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int, optional
            The stage's id. Default value is None.

        Returns
        -------
        list or dict
            A list with all the application's stages' data or a dict with a
            stage's data if profiling is enabled, `None` otherwise.

        """
        if self._enabled:
            if stage_id is None:
                return self._request("/{}/stages".format(app_id))
            else:
                return self._request("/{}/stages/{}".format(app_id, stage_id))

    def request_stageattempt(self, app_id, stage_id, stageattempt_id):
        """Request an application's stage attempt's data.

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
            A dict with an application's stage attempt's data if profiling is enabled,
            `None` otherwise.

        """
        if self._enabled:
            return self._request(
                "/{}/stages/{}/{}".format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasksummary(
            self, app_id, stage_id, stageattempt_id):
        """Request the task summary of a stage attempt.

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
            `None` otherwise.

        """
        if self._enabled:
            return self._request(
                "/{}/stages/{}/{}/taskSummary".format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasklist(self, app_id, stage_id, stageattempt_id):
        """Request the task list of a stage attempt.

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
            `None` otherwise.

        """
        if self._enabled:
            return self._request(
                "/{}/stages/{}/{}/taskList".format(app_id, stage_id, stageattempt_id))

    def request_executors(self, app_id):
        """Request all the application's active executors' data.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        list
            A list with all the application's active executors' data if profiling is enabled,
            `None` otherwise.

        """
        if self._enabled:
            return self._request("/{}/executors".format(app_id))

    def request_allexecutors(self, app_id):
        """Request all the application's executors' data.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        list
            A list with all the application's executors' data if profiling is enabled,
            `None` otherwise.

        """
        if self._enabled:
            return self._request("/{}/allexecutors".format(app_id))

    def request_rdd(self, app_id, rdd_id=None):
        """Request all the application's RDDs' data.

        Parameters
        ----------
        app_id : int
            The application's id.
        rdd_id : int, optional
            The RDD's id. Default value is None.

        Returns
        -------
        list or dict
            A list with all the application's RDDs' data or a dict with the
            RDD's data if profiling is enabled, `None` otherwise.

        """
        if self._enabled:
            if rdd_id is None:
                return self._request("/{}/storage/rdd".format(app_id))
            else:
                return self._request(
                    "/{}/storage/rdd/{}".format(app_id, rdd_id))

    def reset(self):
        """Reset the profiler's attributes, in order to get ready for a new profiling round."""
        self._start()

    def log_rdd(self, app_id, rdd_id=None):
        """Log all the application's RDDs' data.

        Parameters
        ----------
        app_id : int
            The application's id.
        rdd_id : int, optional
            The RDD's id. Default value is None.

        """
        data = self.request_rdd(app_id, rdd_id)

        if data is None or len(data) == 0:
            self._logger.info("no stored RDD to have its data logged")
        else:
            if rdd_id is None:
                self._logger.info("logging the application's RDDs' data...")
            else:
                self._logger.info(
                    "logging the application's RDD {}'s data...".format(rdd_id))

            for d in data:
                for k, v in d.items():
                    if k != 'partitions':
                        self._logger.info("{}: {}".format(k, v))

    def log_executors(self, app_id, exec_id=None):
        """Log all the application's active executors' data.

        Parameters
        ----------
        app_id : int
            The application's id.
        exec_id : int, optional
            The executor's id. Default value is None.

        """
        data = self.request_executors(app_id)

        if data is None or len(data) == 0:
            self._logger.info("no active executor to have its data logged")
        else:
            if exec_id is None:
                self._logger.info(
                    "logging the application's active executors' data...")

                for d in data:
                    for k, v in d.items():
                        self._logger.info("{}: {}".format(k, v))
            else:
                ex_profile = None

                for d in data:
                    if exec_id == d['id']:
                        ex_profile = d
                        break

                if ex_profile is None:
                    self._logger.warning(
                        "executor {} is not active".format(exec_id))
                else:
                    for k, v in ex_profile.items():
                        self._logger.info("{}: {}".format(k, v))

    def profile_rdd(self, app_id):
        """Store all the application's RDDs' data.

        Parameters
        ----------
        app_id : int
            The application's id.

        """
        if self._enabled:
            self._logger.info("profiling all the application's RDDs' data...")

            data = self.request_rdd(app_id)

            for d in data:
                if d['id'] not in self._rdd:
                    self._rdd[d['id']] = []

                rdd_profile = self._rdd_metrics()

                self._rdd[d['id']].append(rdd_profile)

                for k, v in d.items():
                    if k in rdd_profile:
                        rdd_profile[k] = v

    def profile_executors(self, app_id):
        """Store all the application's active executors' data.

        Parameters
        ----------
        app_id : int
            The application's id.

        """
        if self._enabled:
            self._logger.info(
                "profiling all the application's active executors' data...")

            data = self.request_executors(app_id)

            if data is not None:
                for d in data:
                    if d['id'] not in self._executors:
                        self._executors[d['id']] = []

                    ex_profile = self._executor_metrics()

                    self._executors[d['id']].append(ex_profile)

                    for k, v in d.items():
                        if k in ex_profile:
                            ex_profile[k] = v

                    if 'memoryMetrics' in d:
                        ex_profile['memoryMetrics.usedOnHeapStorageMemory'] = d['memoryMetrics']['usedOnHeapStorageMemory']
                        ex_profile['memoryMetrics.usedOffHeapStorageMemory'] = d['memoryMetrics']['usedOffHeapStorageMemory']
                        ex_profile['memoryMetrics.totalOnHeapStorageMemory'] = d['memoryMetrics']['totalOnHeapStorageMemory']
                        ex_profile['memoryMetrics.totalOffHeapStorageMemory'] = d['memoryMetrics']['totalOffHeapStorageMemory']

                    if 'peakMemoryMetrics' in d:
                        ex_profile['peakMemoryMetrics.OnHeapExecutionMemory'] = d['peakMemoryMetrics']['OnHeapExecutionMemory']
                        ex_profile['peakMemoryMetrics.OffHeapExecutionMemory'] = d['peakMemoryMetrics']['OffHeapExecutionMemory']
                        ex_profile['peakMemoryMetrics.OnHeapStorageMemory'] = d['peakMemoryMetrics']['OnHeapStorageMemory']
                        ex_profile['peakMemoryMetrics.OffHeapStorageMemory'] = d['peakMemoryMetrics']['OffHeapStorageMemory']
                        ex_profile['peakMemoryMetrics.OnHeapUnifiedMemory'] = d['peakMemoryMetrics']['OnHeapUnifiedMemory']
                        ex_profile['peakMemoryMetrics.OffHeapUnifiedMemory'] = d['peakMemoryMetrics']['OffHeapUnifiedMemory']

    def get_rdd(self, rdd_id=None):
        """Get all the previously profiled RDDs' data.

        Parameters
        ----------
        rdd_id : int, optional
            The RDD's id. Default value is None.

        Returns
        -------
        dict or list
            A dict with all the previously profiled RDDs' data or a list with the RDD's data.

        """
        if len(self._rdd):
            if rdd_id is None:
                return self._rdd.copy()
            else:
                if rdd_id not in self._rdd:
                    self._logger.warning(
                        "no data from RDD {} has been stored yet".format(rdd_id))
                    return []

                return self._rdd[rdd_id]
        else:
            self._logger.warning("no RDD's data has been stored yet")
            return {}

    def get_executors(self, exec_id=None):
        """Get all the previously profiled executors' data.

        Parameters
        ----------
        exec_id : int, optional
            An executor id. Default value is None.

        Returns
        -------
        dict or list
            A dict with all the previously profiled executors' data or a list with the executor's data.

        """
        if len(self._executors):
            if exec_id is None:
                return self._executors.copy()
            else:
                if exec_id not in self._executors:
                    self._logger.warning(
                        "no data from executor {} has been stored yet".format(exec_id))
                    return []

                return self._executors[exec_id]
        else:
            self._logger.warning("no executors' data has been stored yet")
            return self._executors

    def export_rdd(self, path, extension='csv'):
        """Export all the stored RDDs' data.

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
            "exporting all the stored RDDs' data in {} format...".format(extension))

        path = util.append_slash(path) + 'profiling/rdd/'
        util.create_dir(path)

        if len(self._rdd):
            fieldnames = self._rdd_metrics().keys()

            for k, v in self._rdd.items():
                self._export(v,
                             fieldnames,
                             "{}rdd_{}".format(path, k),
                             extension)

            self._logger.info("RDD data successfully exported")
        else:
            self._logger.warning("no RDD data has been stored yet")

    def export_executors(self, path, extension='csv'):
        """Export all stored executors' data.

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

        path = util.append_slash(path) + 'profiling/executors/'
        util.create_dir(path)

        if len(self._executors):
            fieldnames = self._executor_metrics().keys()

            for k, v in self._executors.items():
                self._export(v,
                             fieldnames,
                             "{}executor_{}".format(path, k),
                             extension)

            self._logger.info("executor data successfully exported")
        else:
            self._logger.warning("no executor data has been stored yet")

    def export(self, path, extension='csv'):
        """Export all stored profiling data.

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
