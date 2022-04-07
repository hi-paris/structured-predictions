
class StructuredOutputMixin:
    """Mixin to mark estimators that support structured prediction."""
    def _more_tags(self):
        return {'structured_output': True}

