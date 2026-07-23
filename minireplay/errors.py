class ReplayError(Exception):
    """Base class for expected replay failures."""


class ValidationError(ReplayError):
    """Input or evidence violates a fail-closed contract."""


class MismatchError(ReplayError):
    """Native execution drifted from the recording."""


class InfrastructureError(ReplayError):
    """The host or runtime could not execute a valid attempt."""


class WorkloadComplete(ReplayError):
    """The framework asked for work beyond the end of the recorded window.

    A recording is cut at the sweep's sample boundary, usually mid-episode, so a
    correct replay reaches the end of the recorded work with the framework still
    willing to continue. That request is not drift: the source simply never got to
    make it. It is held forever instead, so the framework can neither observe an
    invented result nor refill a task the recording does not contain.

    This is only benign once every recorded slot is consumed. The same request
    arriving earlier is real drift and stays a hard failure.
    """
