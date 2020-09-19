from sparkquantum import util
from sparkquantum.dtqw.operator import is_operator
from sparkquantum.dtqw.state import is_state
from sparkquantum.math.distribution import is_distribution
from sparkquantum.profiler import Profiler

__all__ = ['QuantumWalkProfiler']


class QuantumWalkProfiler(Profiler):
    """Profile and export the resources consumed by Spark_ for quantum walks.

    .. _Spark:
        https://spark.apache.org

    """

    def __init__(self):
        """Build a quantum walk profiler object."""
        super().__init__()

        self._times = None
        self._operators = None
        self._states = None
        self._distributions = None

        self._start()

    @staticmethod
    def _default_operator():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0,
                'size': 0, 'numElements': 0}

    @staticmethod
    def _default_state():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0,
                'size': 0, 'numElements': 0}

    @staticmethod
    def _default_distribution():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0,
                'size': 0, 'numElements': 0}

    def __str__(self):
        return 'Quantum Walk Profiler configured to request data from {}'.format(
            self._base_url)

    def _start(self):
        super()._start()

        self._times = {}
        self._operators = {}
        self._states = {}
        self._distributions = {}

    def profile_time(self, name, value):
        """Store the execution or building time for a named quantum walk element.

        Parameters
        ----------
        name : str
            A name for the element.
        value : float
            The measured execution or building time of the element.

        """
        if self._enabled:
            self._logger.info("profiling time for '{}'...".format(name))

            self._times[name] = value

    def profile_operator(self, name, operator, time):
        """Store building time and resources information for a named quantum walk operator.

        Parameters
        ----------
        name : str
            A name for the operator.
        operator : :py:class:`sparkquantum.dtqw.operator.Operator`
            The :py:class:`sparkquantum.dtqw.operator.Operator` object.
        time : float
            The measured building time of the operator.

        Returns
        -------
        dict
            The resources information measured for the operator if profiling is enabled, `None` otherwise.

        Raises
        -----
        TypeError
            If `operator` is not a :py:class:`sparkquantum.dtqw.operator.Operator`.

        """
        if self._enabled:
            self._logger.info(
                "profiling operator data for '{}'...".format(name))

            if not is_operator(operator):
                self._logger.error(
                    "'Operator' instance expected, not '{}'".format(type(operator)))
                raise TypeError(
                    "'Operator' instance expected, not '{}'".format(type(operator)))

            if name not in self._operators:
                self._operators[name] = []

            self._operators[name].append(self._default_operator())

            app_id = operator.sc.applicationId
            rdd_id = operator.data.id()
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in self._default_operator():
                        self._operators[name][-1][k] = v

            self._operators[name][-1]['buildingTime'] = time
            self._operators[name][-1]['size'] = operator.size
            self._operators[name][-1]['numElements'] = operator.nelem

            return self._operators[name][-1]

    def profile_state(self, name, state, time):
        """Store building time and resources information for a named quantum walk system state.

        Parameters
        ----------
        name : str
            A name for the state.
        state : :py:class:`sparkquantum.dtqw.state.State`
            The ``nth`` step corresponding state.
        time : float
            The measured building time of the state.

        Returns
        -------
        dict
            The resources information measured for the state if profiling is enabled, `None` otherwise.

        Raises
        -----
        TypeError
            If `state` is not a :py:class:`sparkquantum.dtqw.state.State`.

        """
        if self._enabled:
            self._logger.info(
                "profiling quantum system state data for '{}'...".format(name))

            if not is_state(state):
                self._logger.error(
                    "'State' instance expected, not '{}'".format(type(state)))
                raise TypeError(
                    "'State' instance expected, not '{}'".format(type(state)))

            if name not in self._states:
                self._states[name] = []

            self._states[name].append(self._default_state())

            app_id = state.sc.applicationId
            rdd_id = state.data.id()
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in self._default_state():
                        self._states[name][-1][k] = v

            self._states[name][-1]['buildingTime'] = time
            self._states[name][-1]['size'] = state.size
            self._states[name][-1]['numElements'] = state.nelem

            return self._states[name][-1]

    def profile_distribution(
            self, name, distribution, time):
        """Store building time and resources information for a named measurement (probability distribution).

        Parameters
        ----------
        name : str
            A name for the distribution.
        distribution : :py:class:`sparkquantum.math.distribution.ProbabilityDistribution`
            The probability distribution object.
        time : float
            The measured building time of the probability distribution.

        Returns
        -------
        dict
            The resources information measured for the probability distribution if profiling is enabled, `None` otherwise.

        Raises
        -----
        TypeError
            If `distribution` is not a :py:class:`sparkquantum.math.distribution.ProbabilityDistribution`.

        """
        if self._enabled:
            self._logger.info(
                "profiling probability distribution data for '{}'...".format(name))

            if not is_distribution(distribution):
                self._logger.error(
                    "'ProbabilityDistribution' instance expected, not '{}'".format(type(distribution)))
                raise TypeError(
                    "'ProbabilityDistribution' instance expected, not '{}'".format(type(distribution)))

            if name not in self._distributions:
                self._distributions[name] = []

            self._distributions[name].append(
                self._default_distribution())

            app_id = distribution.sc.applicationId
            rdd_id = distribution.data.id()
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in self._default_distribution():
                        self._distributions[name][-1][k] = v

            self._distributions[name][-1]['buildingTime'] = time
            self._distributions[name][-1]['size'] = distribution.size
            self._distributions[name][-1]['numElements'] = distribution.nelem

            return self._distributions[name][-1]

    def get_times(self, name=None):
        """Get the measured time of all elements or of the named one.

        Parameters
        ----------
        name : str, optional
            A name for the element.

        Returns
        -------
        dict or float
            A dict with the measured time of all elements or the measured time of the named element.

        """
        if len(self._times):
            if name is None:
                return self._times.copy()
            else:
                if name not in self._times:
                    self._logger.warning(
                        "no measurement of time has been done for '{}'".format(name))
                    return {}
                return self._times[name]
        else:
            self._logger.warning("no measurement of time has been done")
            return {}

    def get_operators(self, name=None):
        """Get the resources information of all operators or of the one with the provided name.

        Parameters
        ----------
        name : str, optional
            The name used for a :py:class:`sparkquantum.dtqw.operator.Operator`.

        Returns
        -------
        dict or list
            A dict with the resources information of all operators or a list with the resources information of the named operator.

        """
        if len(self._operators):
            if name is None:
                return self._operators.copy()
            else:
                if name not in self._operators:
                    self._logger.warning(
                        "no resources information for operator '{}'".format(name))
                    return {}
                return self._operators[name]
        else:
            self._logger.warning(
                "no resources information for operators have been obtained")
            return {}

    def get_states(self, name=None):
        """Get the resources information of all states or of the one with the provided name.

        Parameters
        ----------
        name : str, optional
            The name used for a :py:class:`sparkquantum.dtqw.state.State`.

        Returns
        -------
        dict or list
            A dict with the resources information of all states or a list with the resources information of the named state.

        """
        if len(self._states):
            if name is None:
                return self._states.copy()
            else:
                if name not in self._states:
                    self._logger.warning(
                        "no resources information for state '{}'".format(name))
                    return {}
                return self._states[name]
        else:
            self._logger.warning(
                "no resources information for states have been obtained")
            return {}

    def get_distributions(self, name=None):
        """Get the resources information of all probability distributions or
        of the one with the provided name.

        Parameters
        ----------
        name : str, optional
            The name used for a :py:class:`sparkquantum.math.distribution.ProbabilityDistribution`.

        Returns
        -------
        dict or list
            A dict with the resources information of all probability distributions or
            a list of the resources information of the named probability distribution.

        """
        if len(self._distributions):
            if name is None:
                return self._distributions.copy()
            else:
                if name not in self._distributions:
                    self._logger.warning(
                        "no resources information for probability distribution '{}'".format(name))
                    return {}
                return self._distributions[name]
        else:
            self._logger.warning(
                "no resources information for probability distributions have been obtained")
            return {}

    def export_times(self, path, extension='csv'):
        """Export all stored execution and/or building times.

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
        self._logger.info("exporting times in {} format...".format(extension))

        if len(self._times):
            self._export_values([self._times], self._times.keys(),
                                util.append_slash(path) + 'times', extension)

            self._logger.info("times successfully exported")
        else:
            self._logger.warning("no measurement of time has been done")

    def export_operators(self, path, extension='csv'):
        """Export all stored operators' resources.

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
            "exporting operators' resources in {} format...".format(extension))

        if len(self._operators):
            operator = []

            for k, v in self._operators.items():
                for i in v:
                    operator.append(i.copy())
                    operator[-1]['name'] = k

            self._export_values(operator, operator[-1].keys(),
                                util.append_slash(path) + 'operators', extension)

            self._logger.info("operator's resources successfully exported")
        else:
            self._logger.warning(
                "no measurement of operators' resources has been done")

    def export_states(self, path, extension='csv'):
        """Export all stored state' resources.

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
            "exporting states' resources in {} format...".format(extension))

        if len(self._states):
            states = []

            for k, v in self._states.items():
                for i in v:
                    states.append(i.copy())
                    states[-1]['name'] = k

            self._export_values(states, states[-1].keys(),
                                util.append_slash(path) + 'states', extension)

            self._logger.info("states' resources successfully exported")
        else:
            self._logger.warning(
                "no measurement of states' resources has been done")

    def export_distributions(self, path, extension='csv'):
        """Export all stored probability distributions' resources.

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
            "exporting probability distributions' resources in {} format...".format(extension))

        if len(self._distributions):
            distributions = []

            for k, v in self._distributions.items():
                for i in v:
                    distributions.append(i.copy())
                    distributions[-1]['name'] = k

            self._export_values(distributions, distributions[-1].keys(),
                                util.append_slash(path) + 'distributions', extension)

            self._logger.info(
                "probability distributions' resources successfully exported")
        else:
            self._logger.warning(
                "no measurement of probability distributions' resources has been done")

    def export(self, path, extension='csv'):
        """Export all stored profiling information of quantum walks.

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
        super().export(path, extension)

        self.export_times(path, extension)
        self.export_operators(path, extension)
        self.export_states(path, extension)
        self.export_distributions(path, extension)
