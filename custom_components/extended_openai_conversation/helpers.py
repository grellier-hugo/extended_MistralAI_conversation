
import logging
from mistralai.async_client import MistralAsyncClient

from homeassistant.core import HomeAssistant
from homeassistant.helpers.template import Template


from .exceptions import (
    FunctionNotFound,
)


_LOGGER = logging.getLogger(__name__)


def convert_to_template(
    settings,
    template_keys=["data", "event_data", "target", "service"],
    hass: HomeAssistant | None = None,
):
    _convert_to_template(settings, template_keys, hass, [])


def _convert_to_template(settings, template_keys, hass, parents: list[str]):
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)



async def validate_authentication(
    hass: HomeAssistant,
    api_key: str,
    endpoint: str,
    skip_authentication=False,
) -> None:
    if skip_authentication:
        return
    client = MistralAsyncClient(api_key=api_key, endpoint=endpoint)
    await client.list_models()

