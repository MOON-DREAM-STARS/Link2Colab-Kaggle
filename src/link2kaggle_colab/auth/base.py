"""Authentication provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from link2kaggle_colab.core.errors import NotSupportedError
from link2kaggle_colab.core.models import AuthStatus, ProviderName


class AuthProvider(ABC):
    """Reports auth state. ``refresh`` is optional and defaults to not supported."""

    provider: ProviderName

    @abstractmethod
    def status(self) -> AuthStatus:
        """Return the current authentication snapshot."""

    def refresh(self) -> AuthStatus:
        """Refresh credentials. Providers that cannot refresh raise ``NotSupportedError``."""
        if isinstance(self.provider, ProviderName):
            name = self.provider.value
        else:
            name = str(self.provider)
        raise NotSupportedError(
            f"{type(self).__name__} does not support refresh",
            provider=name,
        )
