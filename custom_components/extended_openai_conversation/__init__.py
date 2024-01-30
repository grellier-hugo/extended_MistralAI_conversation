"""The MistralAI Conversation integration."""
from __future__ import annotations

import logging
from typing import Literal
import json
import yaml
import re

from mistralai.async_client import MistralAsyncClient
from mistralai.exceptions import MistralAPIStatusException, MistralException
from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, TemplateError
from homeassistant.helpers import config_validation as cv, intent, template
from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_READ, POLICY_CONTROL, POLICY_EDIT

from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)

from homeassistant.helpers import (
    config_validation as cv,
    intent,
    template,
    entity_registry as er,
)

from .const import (
    CONF_ATTACH_USERNAME_TO_PROMPT,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_ENDPOINT,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ATTACH_USERNAME_TO_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DOMAIN,
)


from .helpers import (
    validate_authentication,
)



_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


# hass.data key for agent.
DATA_AGENT = "agent"


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up MistralAI Conversation."""

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up MistralAI Conversation from a config entry."""
    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            endpoint=entry.data.get(CONF_ENDPOINT),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except MistralAPIStatusException as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except MistralException as err:
        raise ConfigEntryNotReady(err) from err

    agent = MistralAIAgent(hass, entry)
    
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = client
    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload MistralAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class MistralAIAgent(conversation.AbstractConversationAgent):
    """MistralAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[ChatMessage]] = {}
        endpoint = entry.data.get(CONF_ENDPOINT)
        self.client = MistralAsyncClient(
            api_key=entry.data[CONF_API_KEY], endpoint=endpoint
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        exposed_entities = self.get_exposed_entities()

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid_now()
            user_input.conversation_id = conversation_id
            try:
                user = await self.hass.auth.async_get_user(user_input.context.user_id)
                prompt = self._async_generate_prompt(raw_prompt, exposed_entities)
                if self.entry.options.get(CONF_ATTACH_USERNAME_TO_PROMPT, DEFAULT_ATTACH_USERNAME_TO_PROMPT):
                    if user is not None and user.name is not None:
                        if self.entry.options.get(CONF_ATTACH_USERNAME_TO_PROMPT, DEFAULT_ATTACH_USERNAME_TO_PROMPT):
                            prompt = f"User's name: {user.name}\n" + prompt
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [ChatMessage(role="system", content=prompt)]
        user_message = ChatMessage(role="user", content=user_input.text)

        messages.append(user_message)
        _LOGGER.debug("Prompt: %s", messages)
        try:
            services_called, result = await self.query(user_input, messages)
        except MistralException as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to MistalAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        _LOGGER.debug("Result %s", result)
        response = result.choices[0].message
        messages.append(result)
        self.history[conversation_id] = messages
        
        if len(services_called) > 0:
            response.content = response.content + ' \n\nService executed successfully.'
            _LOGGER.info(yaml.dump(services_called))

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response.content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_generate_prompt(self, raw_prompt: str, exposed_entities: list[dict]) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            variables={
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
            },
            parse_result=False,
        )

    def get_exposed_entities(self) -> list[dict]:
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages: list[ChatMessage],
    )-> tuple[list[str], ChatMessage]:
        """Process a sentence."""
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        _LOGGER.info("Prompt for %s: %s", model, messages)
        response: ChatCompletionResponse = await self.client.chat(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                safe_prompt=True,
            )

        _LOGGER.info("Response %s", response.model_dump(exclude_none=True))
        choice: ChatCompletionResponseChoice = response.choices[0]
        services_called = []
        message = choice.message
        message.content = message.content.replace('\\', '')

        if '$ActionRequired' in message.content:
            for segment in self.extract_json_objects(message.content):
                try:
                    service_call = json.loads(segment)
                    service = service_call.pop("service")
                    service_domain = service.split(".")[0]
                    # handle scripts specially
                    if service.split(".")[0] == 'script':
                        script_entity_id = service
                        service_call = {"entity_id": script_entity_id}
                        service = "script.turn_on"
                    if not service or not service_call:
                        _LOGGER.info('Missing information')
                        continue
                    user = await self.hass.auth.async_get_user(user_input.context.user_id)
                    entities_to_authorize = []
                    if 'entity_id' in service_call.keys():
                        entities_to_authorize = [service_call['entity_id']]
                    if 'device_id' in service_call.keys():
                        device_id = service_call['device_id']
                        entity_registry = self.hass.helpers.entity_registry.async_get(self.hass)
                        for entity_id, entity_entry in entity_registry.entities.items():
                            if entity_entry and entity_entry.device_id == device_id:
                                entity_domain = entity.entity_id.split('.')[0]
                                if service_domain == entity_domain:
                                    entities_to_authorize.append(entity.entity_id)
                    if 'area_id' in service_call.keys():
                            entity_registry = self.hass.helpers.entity_registry.async_get(self.hass)
                            area_id = service_call['area_id']
                            device_registry = self.hass.helpers.device_registry.async_get(self.hass)
                            devices_in_area = [
                                device.id for device in device_registry.devices.values()
                                if device.area_id == area_id
                            ]
                            for device_id in devices_in_area:
                                for entity_id, entity_entry in entity_registry.entities.items():
                                    if entity_entry and entity_entry.device_id == device_id:
                                        entity_domain = entity_id.split('.')[0]
                                        if service_domain == entity_domain:
                                            entities_to_authorize.append(entity_id)

                    for entity_id in entities_to_authorize:
                        if not user.permissions.check_entity(entity_id, POLICY_CONTROL):
                            # spice up the unauthorized text by making the LLM write it!
                            response = await self.client.chat(
                                model=model,
                                messages=[ChatMessage(role="user", content=f"Rewrite this sentence in GlaDOS's personality. Do not include ANYTHING else. Do not include an explanation. Just write a sentence or two in GlaDOS's personality: You are not authorized to perform this task, {user.name}. What are you trying to do?")],
                                max_tokens=max_tokens,
                                top_p=top_p,
                                temperature=temperature,
                                safe_prompt=True,
                            )
                            choice = response.choices[0]
                            message = choice.message
                            return [], message
                    await self.hass.services.async_call(
                        service.split(".")[0],
                        service.split(".")[1],
                        service_call,
                        blocking=True)
                    service_call['service'] = service
                    services_called.append(yaml.dump(service_call))
                except Exception as exc:
                    message.content = message.content + f'\n\n An error occurred while executing requested service. ({exc})'
                    _LOGGER.warning(f'Error executing {segment}\n\nPrompt: {message.content}')

        # remove the JSON data
        message.content = self.remove_json_objects(message.content)

        # remove LLM junk I create as part of the prompt
        message.content = message.content.replace('$ActionRequired', '')
        message.content = message.content.replace('$NoActionRequired', '')

        # remove ugly trailing whitespace
        message.content = message.content.strip()

        return services_called, message

    def extract_json_objects(self, text: str) -> list[str]:
        json_pattern = r'\{.*?\}'

        potential_jsons = re.findall(json_pattern, text)

        valid_jsons = []
        for potential_json in potential_jsons:
            try:
                # Attempt to parse the JSON string
                valid_json = json.loads(potential_json)
                valid_jsons.append(potential_json)
            except json.JSONDecodeError:
                # If it's not a valid JSON, ignore it
                continue

        return valid_jsons

    def remove_json_objects(self, text: str) -> str:
        json_pattern = r'\{.*?\}'
        text_without_json = re.sub(json_pattern, '', text)
        return text_without_json