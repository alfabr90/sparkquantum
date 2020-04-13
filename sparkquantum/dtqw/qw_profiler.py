from sparkquantum.utils.profiler import Profiler
from sparkquantum.dtqw.operator import is_operator
from sparkquantum.dtqw.state import is_state
from sparkquantum.math.statistics.pdf import is_pdf

__all__ = ['QuantumWalkProfiler']


class QuantumWalkProfiler(Profiler):
    """Profile and export the resources consumed by Spark_ for quantum walks.

    .. _Spark:
        https://spark.apache.org

    """

    def __init__(self, base_url='http://localhost:4040/api/v1/'):
        """Build a quantum walk profiler object.

        Parameters
        ----------
        base_url: str, optional
            The base URL for getting information about the consumed resources. Default value is http://localhost:4040/api/v1/.

        """
        super().__init__(base_url)

        self._operators = None
        self._states = None
        self._pdfs = None

    @staticmethod
    def _default_operator():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0}

    @staticmethod
    def _default_state():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0,
                'numElements': 0, 'numNonzeroElements': 0}

    @staticmethod
    def _default_pdf():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0,
                'numElements': 0, 'numNonzeroElements': 0}

    def start(self):
        """Reset the profiler attributes to get info for a new profiling round."""
        super().start()

        self._operators = {}
        self._states = {}
        self._pdfs = {}

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
            The resources information measured for the operator.

        Raises
        -----
        TypeError
            If `operator` is not a :py:class:`sparkquantum.dtqw.operator.Operator`.

        """
        if self._logger is not None:
            self._logger.info(
                "profiling operator data for '{}'...".format(name))

        if not is_operator(operator):
            if self._logger is not None:
                self._logger.error(
                    "'Operator' instance expected, not '{}'".format(type(operator)))
            raise TypeError(
                "'Operator' instance expected, not '{}'".format(type(operator)))

        if name not in self._operators:
            self._operators[name] = []

        self._operators[name].append(self._default_operator())

        app_id = operator.spark_context.applicationId
        rdd_id = operator.data.id()
        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._default_operator():
                    self._operators[name][-1][k] = v

        self._operators[name][-1]['buildingTime'] = time
        self._operators[name][-1]['numElements'] = operator.num_elements
        self._operators[name][-1]['numNonzeroElements'] = operator.num_nonzero_elements

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
            The resources information measured for the state.

        Raises
        -----
        TypeError
            If `state` is not a :py:class:`sparkquantum.dtqw.state.State`.

        """
        if self._logger is not None:
            self._logger.info(
                "profiling quantum system state data for '{}'...".format(name))

        if not is_state(state):
            if self._logger is not None:
                self._logger.error(
                    "'State' instance expected, not '{}'".format(type(state)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(state)))

        if name not in self._states:
            self._states[name] = []

        self._states[name].append(self._default_state())

        app_id = state.spark_context.applicationId
        rdd_id = state.data.id()
        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._default_state():
                    self._states[name][-1][k] = v

        self._states[name][-1]['buildingTime'] = time
        self._states[name][-1]['numElements'] = state.num_elements
        self._states[name][-1]['numNonzeroElements'] = state.num_nonzero_elements

        return self._states[name][-1]

    def profile_pdf(self, name, pdf, time):
        """Store building time and resources information for a named measurement (PDF).

        Parameters
        ----------
        name : str
            A name for the pdf.
        pdf : :py:class:`sparkquantum.math.statistics.pdf.PDF`
            The pdf object.
        time : float
            The measured building time of the pdf.

        Returns
        -------
        dict
            The resources information measured for the pdf.

        Raises
        -----
        TypeError
            If `pdf` is not a :py:class:`sparkquantum.math.statistics.pdf.PDF`.

        """
        if self._logger is not None:
            self._logger.info("profiling PDF data for '{}'...".format(name))

        if not is_pdf(pdf):
            if self._logger is not None:
                self._logger.error(
                    "'PDF' instance expected, not '{}'".format(type(pdf)))
            raise TypeError(
                "'PDF' instance expected, not '{}'".format(type(pdf)))

        if name not in self._pdfs:
            self._pdfs[name] = []

        self._pdfs[name].append(self._default_pdf())

        app_id = pdf.spark_context.applicationId
        rdd_id = pdf.data.id()
        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._default_pdf():
                    self._pdfs[name][-1][k] = v

        self._pdfs[name][-1]['buildingTime'] = time
        self._pdfs[name][-1]['numElements'] = pdf.num_elements
        self._pdfs[name][-1]['numNonzeroElements'] = pdf.num_nonzero_elements

        return self._pdfs[name][-1]

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
                    if self._logger is not None:
                        self._logger.warning(
                            "no resources information for operator '{}'".format(name))
                    return {}
                return self._operators[name]
        else:
            if self._logger is not None:
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
                    if self._logger is not None:
                        self._logger.warning(
                            "no resources information for state '{}'".format(name))
                    return {}
                return self._states[name]
        else:
            if self._logger is not None:
                self._logger.warning(
                    "no resources information for states have been obtained")
            return {}

    def get_pdfs(self, name=None):
        """Get the resources information of all PDFs or of the one with the provided name.

        Parameters
        ----------
        name : str, optional
            The name used for a :py:class:`sparkquantum.math.statistics.pdf.PDF`.

        Returns
        -------
        dict or list
            A dict with the resources information of all PDFs or a list of the resources information of the named PDF.

        """
        if len(self._pdfs):
            if name is None:
                return self._pdfs.copy()
            else:
                if name not in self._pdfs:
                    if self._logger is not None:
                        self._logger.warning(
                            "no resources information for pdf '{}'".format(name))
                    return {}
                return self._pdfs[name]
        else:
            if self._logger is not None:
                self._logger.warning(
                    "no resources information for pdfs have been obtained")
            return {}

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
        if self._logger is not None:
            self._logger.info(
                "exporting operators' resources in {} format...".format(extension))

        if len(self._operators):
            operator = []

            for k, v in self._operators.items():
                for i in v:
                    operator.append(i.copy())
                    operator[-1]['name'] = k

            self._export_values(
                operator, operator[-1].keys(), path + 'operators', extension)

            if self._logger is not None:
                self._logger.info("operator's resources successfully exported")
        else:
            if self._logger is not None:
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
        if self._logger is not None:
            self._logger.info(
                "exporting states' resources in {} format...".format(extension))

        if len(self._states):
            states = []

            for k, v in self._states.items():
                for i in v:
                    states.append(i.copy())
                    states[-1]['name'] = k

            self._export_values(
                states, states[-1].keys(), path + 'states', extension)

            if self._logger is not None:
                self._logger.info("states' resources successfully exported")
        else:
            if self._logger is not None:
                self._logger.warning(
                    "no measurement of states' resources has been done")

    def export_pdfs(self, path, extension='csv'):
        """Export all stored pdfs' resources.

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
        if self._logger is not None:
            self._logger.info(
                "exporting pdfs' resources in {} format...".format(extension))

        if len(self._operators):
            pdfs = []

            for k, v in self._pdfs.items():
                for i in v:
                    pdfs.append(i.copy())
                    pdfs[-1]['name'] = k

            self._export_values(
                pdfs, pdfs[-1].keys(), path + 'pdfs', extension)

            if self._logger is not None:
                self._logger.info("pdfs' resources successfully exported")
        else:
            if self._logger is not None:
                self._logger.warning(
                    "no measurement of pdfs' resources has been done")

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

        self.export_operators(path, extension)
        self.export_states(path, extension)
        self.export_pdfs(path, extension)
