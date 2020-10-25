from sparkquantum import util
from sparkquantum.dtqw.operator import is_operator
from sparkquantum.dtqw.state import is_state
from sparkquantum.math.distribution import is_distribution
from sparkquantum.profiler import Profiler

__all__ = ['QuantumWalkProfiler', 'get_profiler']


class QuantumWalkProfiler(Profiler):
    """Class for profiling quantum walks."""

    def __init__(self):
        """Build a profiler object for quantum walks."""
        super().__init__()

        self._operators = None
        self._states = None
        self._distributions = None

        self._start()

    def _operator_data(self):
        return {'memoryUsed': 0,
                'diskUsed': 0,
                'size': 0,
                'numElements': 0,
                'buildingTime': 0.0}

    def _state_data(self):
        return {'memoryUsed': 0,
                'diskUsed': 0,
                'size': 0,
                'numElements': 0,
                'buildingTime': 0.0}

    def _distribution_data(self):
        return {'memoryUsed': 0,
                'diskUsed': 0,
                'size': 0,
                'numElements': 0,
                'buildingTime': 0.0}

    def __str__(self):
        return 'Quantum Walk Profiler configured to request data from {}'.format(
            self._base_url)

    def _start(self):
        super()._start()

        self._operators = {}
        self._states = {}
        self._distributions = {}

    def profile_operator(self, name, operator, time):
        """Store a named quantum walk operator's data.

        Parameters
        ----------
        name : str
            The operator's name.
        operator : :py:class:`sparkquantum.dtqw.operator.Operator`
            The :py:class:`sparkquantum.dtqw.operator.Operator` object.
        time : float
            The building time of the operator.

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

            op_profile = self._operator_data()

            app_id = operator.sc.applicationId
            rdd_id = operator.data.id()
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in op_profile:
                        op_profile[k] = v

            op_profile['buildingTime'] = time
            op_profile['size'] = operator.size
            op_profile['numElements'] = operator.nelem

            self._operators[name].append(op_profile)

    def profile_state(self, name, state, time):
        """Store a named quantum walk state's data.

        Parameters
        ----------
        name : str
            The state's name.
        state : :py:class:`sparkquantum.dtqw.state.State`
            The ``nth`` step corresponding state.
        time : float
            The building time of the state.

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

            st_profile = self._state_data()

            app_id = state.sc.applicationId
            rdd_id = state.data.id()
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in st_profile:
                        st_profile[k] = v

            st_profile['buildingTime'] = time
            st_profile['size'] = state.size
            st_profile['numElements'] = state.nelem

            self._states[name].append(st_profile)

    def profile_distribution(
            self, name, distribution, time):
        """Store a named quantum walk probability distribution's data.

        Parameters
        ----------
        name : str
            The distribution's name.
        distribution : :py:class:`sparkquantum.math.distribution.ProbabilityDistribution`
            The probability distribution object.
        time : float
            The building time of the probability distribution.

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

            dist_profile = self._distribution_data()

            app_id = distribution.sc.applicationId
            rdd_id = distribution.data.id()
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in dist_profile:
                        dist_profile[k] = v

            dist_profile['buildingTime'] = time
            dist_profile['size'] = distribution.size
            dist_profile['numElements'] = distribution.nelem

            self._distributions[name].append(dist_profile)

    def get_operators(self, name=None):
        """Get all the previously profiled operators' data.

        Parameters
        ----------
        name : str, optional
            The operator's name. Default value is None.

        Returns
        -------
        dict or list
            A dict with all the previously profiled operators' data or a list with the operator's data.

        """
        if len(self._operators):
            if name is None:
                return self._operators.copy()
            else:
                if name not in self._operators:
                    self._logger.warning(
                        "no resources information for operator '{}'".format(name))
                    return []

                return self._operators[name]
        else:
            self._logger.warning(
                "no resources information for operators have been obtained")
            return {}

    def get_states(self, name=None):
        """Get all the previously profiled states' data.

        Parameters
        ----------
        name : str, optional
            The state's data. Default value is None.

        Returns
        -------
        dict or list
            A dict with all the previously profiled states' data or a list with the state's data.

        """
        if len(self._states):
            if name is None:
                return self._states.copy()
            else:
                if name not in self._states:
                    self._logger.warning(
                        "no resources information for state '{}'".format(name))
                    return []

                return self._states[name]
        else:
            self._logger.warning(
                "no resources information for states have been obtained")
            return {}

    def get_distributions(self, name=None):
        """Get all the previously profiled probability distributions' data.

        Parameters
        ----------
        name : str, optional
            The distribution's data. Default value is None.

        Returns
        -------
        dict or list
            A dict with all the previously profiled probability distributions' data or
            a list with the probability distribution's data.

        """
        if len(self._distributions):
            if name is None:
                return self._distributions.copy()
            else:
                if name not in self._distributions:
                    self._logger.warning(
                        "no resources information for probability distribution '{}'".format(name))
                    return []

                return self._distributions[name]
        else:
            self._logger.warning(
                "no resources information for probability distributions have been obtained")
            return {}

    def export_operators(self, path, extension='csv'):
        """Export all stored operators' data.

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
            "exporting all the stored operators' data in {} format...".format(extension))

        path = util.append_slash(path) + 'profiling/'
        util.create_dir(path)

        if len(self._operators):
            operators = []

            for k, v in self._operators.items():
                for i in v:
                    operators.append(i.copy())
                    operators[-1]['name'] = k

            self._export(operators,
                         operators[-1].keys(),
                         path + 'operators',
                         extension)

            self._logger.info("operator data successfully exported")
        else:
            self._logger.warning("no operator data has been stored yet")

    def export_states(self, path, extension='csv'):
        """Export all stored states' data.

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
            "exporting all the stored states' data in {} format...".format(extension))

        path = util.append_slash(path) + 'profiling/'
        util.create_dir(path)

        if len(self._states):
            states = []

            for k, v in self._states.items():
                for i in v:
                    states.append(i.copy())
                    states[-1]['name'] = k

            self._export(states, states[-1].keys(), path + 'states', extension)

            self._logger.info("state data successfully exported")
        else:
            self._logger.warning("no state data has been stored yet")

    def export_distributions(self, path, extension='csv'):
        """Export all stored probability distributions' data.

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
            "exporting all the stored distributions' data in {} format...".format(extension))

        path = util.append_slash(path) + 'profiling/'
        util.create_dir(path)

        if len(self._distributions):
            distributions = []

            for k, v in self._distributions.items():
                for i in v:
                    distributions.append(i.copy())
                    distributions[-1]['name'] = k

            self._export(distributions,
                         distributions[-1].keys(),
                         path + 'distributions',
                         extension)

            self._logger.info("distribution data successfully exported")
        else:
            self._logger.warning("no distribution data has been stored yet")

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
        super().export(path, extension)

        self.export_operators(path, extension)
        self.export_states(path, extension)
        self.export_distributions(path, extension)


_profiler = None


def get_profiler():
    """Get the profiler instance of this module following the singleton pattern."""
    global _profiler

    if _profiler is None:
        _profiler = QuantumWalkProfiler()

    return _profiler
