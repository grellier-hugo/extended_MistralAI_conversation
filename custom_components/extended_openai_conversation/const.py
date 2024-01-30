"""Constants for the MistralAI Conversation integration."""

DOMAIN = "mistralai_conversation"
DEFAULT_NAME = "Extended OpenAI Conversation"
CONF_ENDPOINT = "endpoint"
DEFAULT_CONF_ENDPOINT = "https://api.mistral.ai"
CONF_SKIP_AUTHENTICATION = "skip_authentication"
DEFAULT_SKIP_AUTHENTICATION = False
EVENT_AUTOMATION_REGISTERED = "automation_registered_via_extended_mistralai_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """This smart home is controlled by Home Assistant.

An overview of the areas and the devices in this smart home:
{%- for area in areas() %}
  {%- set area_info = namespace(printed=false) %}
  {%- for device in area_devices(area) -%}
    {%- if not device_attr(device, "disabled_by") and not device_attr(device, "entry_type") and device_attr(device, "name") %}
      {%- if not area_info.printed %}

{{ area_name(area) }}:
        {%- set area_info.printed = true %}
      {%- endif %}
- {{ device_attr(device, "name") }}{% if device_attr(device, "model") and (device_attr(device, "model") | string) not in (device_attr(device, "name") | string) %} ({{ device_attr(device, "model") }}){% endif %}
    {%- endif %}
  {%- endfor %}
{%- endfor %}

Answer the user's questions about the world truthfully.

If the user wants to control a device, reject the request and suggest using the Home Assistant app.
"""

CONF_UNAUTHORIZE_PROMPT = "unauthorize_prompt"
DEFAULT_UNAUTHORIZE_PROMPT = """Rewrite this sentence in GlaDOS's personality. Do not include ANYTHING else. Do not include an explanation. Just write a sentence or two in GlaDOS's personality: "
You are not authorized to perform this task, {%- user.name %}.
What are you trying to do?"
"""
CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "mistral-tiny"
CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.5
CONF_ATTACH_USERNAME_TO_PROMPT = "attach_username_to_prompt"
DEFAULT_ATTACH_USERNAME_TO_PROMPT = False
SERVICE_QUERY_IMAGE = "query_image"
