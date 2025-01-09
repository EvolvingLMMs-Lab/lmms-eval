from datetime import timedelta
import functools
import logging
import math
import random
import ssl
from geopy.adapters import (
    RequestsAdapter,
    RequestsHTTPAdapter,
    RequestsHTTPWithSSLContextAdapter,
    requests_available,
    _normalize_proxies,
)
from geopy.distance import distance
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests_cache


error_logger = logging.getLogger("errorLogger")


class CachedRequestsAdapter(RequestsAdapter):
    """The adapter which uses `requests`_ library.

    .. _requests: https://requests.readthedocs.io

    `requests` supports keep-alives, retries, persists Cookies,
    allows response compression and uses HTTP/1.1 [currently].

    ``requests`` package must be installed in order to use this adapter.

    The requests' ``trust_env`` value is set to false, meaning that
    environment doesn't affect the requests' configuration.
    The ``ssl_context`` and ``proxies`` settings can be used for configuration.

    .. versionchanged:: 2.4
        This adapter used to use the `certifi` CA bundle by default,
        if an ``ssl_context`` wasn't provided explicitly. This has been
        changed to use the system CA store by default.
    """

    is_available = requests_available

    def __init__(
        self,
        *,
        proxies,
        ssl_context,
        pool_connections=10,
        pool_maxsize=10,
        max_retries=2,
        pool_block=False,
    ):
        if not requests_available:
            raise ImportError(
                "`requests` must be installed in order to use RequestsAdapter. "
                "If you have installed geopy via pip, you may use "
                "this command to install requests: "
                '`pip install "geopy[requests]"`.'
            )
        proxies = _normalize_proxies(proxies)
        if ssl_context is None:
            # By default requests uses CA bundle from `certifi` package.
            # This is typically overridden with the `REQUESTS_CA_BUNDLE`
            # environment variable. However, trust_env is disabled
            # below to turn off the requests-specific logic of proxy
            # servers configuration, which is re-implemented in geopy
            # so that it's similar between different Adapters implementations.
            #
            # Here, in order to align the adapter's behavior with
            # the default URLLibAdapter, we explicitly pass an ssl context,
            # which would be initialized with the system's CA store
            # rather than the certifi's bundle requests uses by default.
            #
            # See also https://github.com/geopy/geopy/issues/546
            ssl_context = ssl.create_default_context()
        super().__init__(proxies=proxies, ssl_context=ssl_context)

        self.session = requests_cache.CachedSession(
            backend="sqlite", expire_after=timedelta(days=30)
        )
        self.session.trust_env = False  # don't use system proxies
        self.session.proxies = proxies

        self.session.mount(
            "http://",
            RequestsHTTPAdapter(
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
                max_retries=max_retries,
                pool_block=pool_block,
            ),
        )
        self.session.mount(
            "https://",
            RequestsHTTPWithSSLContextAdapter(
                ssl_context=ssl_context,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
                max_retries=max_retries,
                pool_block=pool_block,
            ),
        )


USER_AGENT_SUFFIX = hex(random.getrandbits(128))[2:]
geolocator = Nominatim(
    user_agent=f"vlm-mega-benchmark_{USER_AGENT_SUFFIX}",
    adapter_factory=CachedRequestsAdapter,
)


def calculate_proximity_score(guess_coords, actual_coords, k=100):
    """Calculate the proximity score based on the location.

    Exponentially decreases depending on the distance.

    Args:
        guess_coords (float, float): The longitude and latitude of the guessed coordinates.
        actual_coords (float, float): The longitude and latitude of the actual coordinates.
        k (numbers.Number): The threshold (in km) at which we get a score of 0.5.
    """
    dist = distance(guess_coords, actual_coords).km
    proximity_score = math.exp(-dist / k)
    return proximity_score


GEOLOCATION_TIMEOUT = 1
MAX_RETRIES = 30


geocode = RateLimiter(
    geolocator.geocode, min_delay_seconds=GEOLOCATION_TIMEOUT, max_retries=MAX_RETRIES
)


@functools.cache
def try_geolocate(query):
    """Try to look up the location."""
    location = geocode(query)
    if location is None:
        error_logger.error(
            f"Geolocation API request failed due to timeout: exceeded {MAX_RETRIES} retries!"
        )
    return location


def location_to_coords(
    country: str, province_or_state: str, municipality: str
) -> tuple[float, float] | None:
    if country == "" or province_or_state == "" or municipality == "":
        return None
    """Convert the location to longitude and latitude."""
    location = geolocator.geocode(
        query={"country": country, "state": province_or_state, "city": municipality}
    )
    if location is not None:
        return (location.latitude, location.longitude)
    # Try searching without the province/state, as it can be non-standard for some questions
    location = geolocator.geocode(query={"country": country, "city": municipality})
    if location is None:
        return None
    return (location.latitude, location.longitude)


class GeoProximityLocationDict:
    """Return a score based on the distance between two locations."""

    @classmethod
    def match(cls, responses, targets) -> float:
        """Return a score based on how far two targets are away from each other,
        where each field is a dict with the following schema:
        {
            country: str,
            province_or_state: str,
            municipality: str
        }
        """
        try:
            guess_coords = location_to_coords(**responses)
        except:
            return 0

        if guess_coords is None:
            error_logger.error(
                f"GeoProximityLocationDict: could not load co-ordinates for {responses=}"
            )
            return 0
        actual_coords = location_to_coords(**targets)
        if actual_coords is None:
            error_logger.error(
                f"GeoProximityLocationDict: could not load co-ordinates for {targets=}"
            )
            return 0

        return calculate_proximity_score(guess_coords, actual_coords)
