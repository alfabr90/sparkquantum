from sparkquantum import constants
from sparkquantum.dtqw.interaction.interaction import Interaction

__all__ = ['Collision']


class Collision(Interaction):
    """Top-level class that represents interaction between particles during collisions."""

    def __init__(self):
        """Build an interaction object during collisions."""
        super().__init__()

    def __str__(self):
        return 'Collision interaction'

    def create_operator(self, mesh, particles,
                        repr_format=constants.StateRepresentationFormatCoinPosition):
        """Build the interaction operator for a quantum walk.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError
