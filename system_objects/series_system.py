from system_objects.base_system import BaseSystem


class SeriesSystem(BaseSystem):
    """A system where components are arranged in series."""
    # self.name = name
    # self.components = components

    

    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        if all(component.state == 1 for component in self.components):
        return 1  # "working"
    return 0  # "failed"
