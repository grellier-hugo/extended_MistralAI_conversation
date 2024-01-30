# MistralAI Conversation
This is custom component of Home Assistant.

Derived from [MistralAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/).


## Installation
1. Install via HACS or by copying `MistralAI Conversation` folder into `<config directory>/custom_components`
2. Restart Home Assistant
3. Go to Settings > Devices & Services.
4. In the bottom right corner, select the Add Integration button.
5. Follow the instructions on screen to complete the setup (API Key is required).
    - [Generating an API Key](https://console.mistral.ai/)
    - Specify "Base Url" if using MistralAI compatible servers like LocalAI, otherwise leave as it is.
6. Go to Settings > [Voice Assistants](https://my.home-assistant.io/redirect/voice_assistants/).
7. Click to edit Assistant (named "Home Assistant" by default).
8. Select "MistralAI Conversation" from "Conversation agent" tab.

## Preparation
After installed, you need to expose entities from "http://{your-home-assistant}/config/voice-assistants/expose".

## Configuration
### Options
By clicking a button from Edit Assist, Options can be customized.<br/>
Options include [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) options and one new options. 

- `Attach Username`: Pass the active user's name (if applicable) to MistralAI via the message payload. Currently, this only applies to conversations through the UI or REST API.

## Practical Usage
See more practical [examples](https://github.com/jekalmin/extended_openai_conversation/tree/main/examples).

## Logging
In order to monitor logs of API requests and responses, add following config to `configuration.yaml` file

```yaml
logger:
  logs:
    custom_components.extended_openai_conversation: info
```
