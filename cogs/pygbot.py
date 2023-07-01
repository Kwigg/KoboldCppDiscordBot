import re
import json
import requests
import discord
from discord import app_commands
from discord.ext import commands
import os
import sentencepiece

# configuration settings for the api
model_config = {
    # These don't do anything yet
    "use_story": False,
    "use_authors_note": False,
    "use_world_info": False,
    "use_memory": False,
    # LLaMA tuned params
    "max_context_length": 2048,
    "max_length": 150,
    "rep_pen": 1.19,
    "rep_pen_range": 1024,
    "rep_pen_slope": 0.9,
    "temperature": 0.79,
    "tfs": 0.95,
    "top_a": 0,
    "top_k": 0,
    "top_p": 0.9,
    "typical": 1,
    "n": 1,
    "sampler_order": [6, 0, 1, 2, 3, 4, 5],
    "stop_sequence": ["\<START\>", "<START>", "<STOP>", "<END>", "User:"]
}

seen_users = []

def embedder(msg):
    embed = discord.Embed(description=f"{msg}", color=0x9C84EF)
    return embed

def count_tokens_in_string(eval_string):
    sp = sentencepiece.SentencePieceProcessor(model_file='tokenizer_model/tokenizer.model')
    tokens = sp.encode_as_ids(eval_string)
    return len(tokens)

class Chatbot:
    def __init__(self, char_filename, bot):
        self.prompt = None
        self.endpoint = bot.endpoint
        # Send a PUT request to modify the settings
        # This does not work with koboldcpp.
        # TavernAI instead embeds the config into every api call
        # requests.put(f"{self.endpoint}/config", json=model_config)
        # read character data from JSON file
        with open(char_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char_name = data["char_name"]
            self.char_persona = data["char_persona"]
            self.char_greeting = data["char_greeting"]
            if "world_scenario" in data:
                self.world_scenario = data["world_scenario"]
            self.example_dialogue = data["example_dialogue"]
            if "personality" in data:
                self.personality = data["personality"]

        self.char_persona = self.char_persona.replace("{{char}}", self.char_name)
        self.char_greeting = self.char_greeting.replace("{{char}}", self.char_name)
        self.world_scenario = self.world_scenario.replace("{{char}}", self.char_name)
        self.example_dialogue = self.example_dialogue.replace("{{char}}", self.char_name)
        self.personality = self.personality.replace("{{char}}", self.char_name)

        # initialize conversation history and character information
        self.convo_filename = None
        self.character_info = f"{self.char_name}'s Persona: {self.char_persona}\n"
        if self.personality is not None:
            self.character_info += (f"Description of {self.char_name}: {self.personality}\n")
        if self.world_scenario is not None:
            self.character_info += f"Scenario: {self.world_scenario}\n<START>"

        self.char_info_tokens = count_tokens_in_string(self.character_info)
        self.conversation_queue = []
        self.conversation_queue_total_token = 0

    def dequeue_item_and_update_token_count(self):
        line = self.conversation_queue.pop()
        self.conversation_queue_total_token -= line[1]

    def remove_item_at_index_and_update_token_count(self, index):
        self.conversation_queue_total_token -= self.conversation_queue[index][1]
        del self.conversation_queue[index]

    def extract_prompt_out_of_queue(self):
        prompt_string = ""
        for line in self.conversation_queue:
            prompt_string = line[0] + "\n" + prompt_string # The queue is in reverse, with the last element being the oldest
        return prompt_string

    def push_lines_to_queue_respecting_context_limits(self, in_strings):
        print(in_strings)
        for input_line in in_strings: 
            string_token_count = count_tokens_in_string(input_line)
            # Need to clear items until our new string fits
            while (self.conversation_queue_total_token + self.char_info_tokens + string_token_count) > (model_config["max_context_length"] - model_config["max_length"]):
                self.dequeue_item_and_update_token_count()

            self.conversation_queue.insert(0, (input_line, string_token_count))
            self.conversation_queue_total_token += string_token_count
    def push_single_line_to_queue_respecting_context_limits(self, in_string):
        self.push_lines_to_queue_respecting_context_limits([in_string])

    async def set_convo_filename(self, convo_filename):
        # set the conversation filename and load conversation history from file
        self.convo_filename = convo_filename
        if not os.path.isfile(convo_filename):
            await self.reset_convo_file()
            return
        with open(convo_filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.push_lines_to_queue_respecting_context_limits(lines)
        
    async def reset_convo_file(self):
        print(self.convo_filename)
        # set the conversation filename and load conversation history from file
        if not self.convo_filename:
            return False
        start_dialogue = ""
        if self.example_dialogue is not None:
            start_dialogue = "Example Dialogue: " + self.example_dialogue + "\n"
        start_dialogue += "<START>\n"
        with open(self.convo_filename, "w", encoding="utf-8") as f:
            f.write(start_dialogue)
        self.push_lines_to_queue_respecting_context_limits(start_dialogue.splitlines())
        return True

    async def send_prompt_and_parse_result(self, message):
        print(self.conversation_queue)
        print(message)
        if message is not None:
            if message.author.nick not in seen_users:
                seen_users.append(message.author.nick)
                if message.author.nick + ":" not in model_config["stop_sequence"]:
                    model_config["stop_sequence"].append(message.author.nick + ":")
                print(seen_users)
        message_prompt = {
            "prompt": self.character_info + "\n" + 
            "".join(self.extract_prompt_out_of_queue()) +
            f"{self.char_name}:",
        }
        combined_prompt = {**message_prompt, **model_config}
        print(combined_prompt)
        response = requests.post(f"{self.endpoint}/api/v1/generate", json=combined_prompt)
        response_text = ""
        # check if the request was successful
        if response.status_code == 200:
            # Get the results from the response
            results = response.json()["results"]
            response_list = [line for line in results[0]["text"][1:].split("\n")]
            result = [response_list[0]]
            for item in response_list[1:]:
                if self.char_name in item:
                    result.append(item)
                else:
                    break
            new_list = [item.replace(self.char_name + ": ", "\n") for item in result]
            response_text = "".join(new_list)

            for user in seen_users:
                response_text = response_text.split(user + ":")[0]

        return (response, response_text)

    async def save_conversation(self, message, message_content):
        self.push_single_line_to_queue_respecting_context_limits(f"{message.author.nick}: {message_content}")
        # send a post request to the API endpoint
        (response, response_text) = await self.send_prompt_and_parse_result(message)
        # check if the request was successful
        if response.status_code == 200:
            # add bot response to conversation history
            self.push_single_line_to_queue_respecting_context_limits(f"{self.char_name}: {response_text}")
            with open(self.convo_filename, "a", encoding="utf-8") as f:
                f.write(f"{message.author.nick}: {message_content}\n")
                f.write(f"{self.char_name}: {response_text}\n")  # add a separator between

            return response_text

    async def follow_up(self):
        (response, response_text) = await self.send_prompt_and_parse_result(None)
        # check if the request was successful
        if response.status_code == 200:
            self.push_single_line_to_queue_respecting_context_limits(f"{self.char_name}: {response_text}")
            with open(self.convo_filename, "a", encoding="utf-8") as f:
                f.write(
                    f"{self.char_name}: {response_text}\n"
                )  # add a separator between
            return response_text


class ChatbotCog(commands.Cog, name="chatbot"):
    def __init__(self, bot):
        self.bot = bot
        self.chatlog_dir = bot.chatlog_dir
        self.chatbot = Chatbot("chardata.json", bot)

        # create chatlog directory if it doesn't exist
        if not os.path.exists(self.chatlog_dir):
            os.makedirs(self.chatlog_dir)

    # converts user ids and emoji ids
    async def replace_user_mentions(self, content):
        user_ids = re.findall(r"<@(\d+)>", content)
        for user_id in user_ids:
            user = await self.bot.fetch_user(int(user_id))
            if user:
                display_name = user.display_name
                content = content.replace(f"<@{user_id}>", display_name)

        emojis = re.findall(r"<:[^:]+:(\d+)>", content)
        for emoji_id in emojis:
            if ":" in content:
                emoji_name = content.split(":")[1]
                content = content.replace(f"<:{emoji_name}:{emoji_id}>", f":{emoji_name}:")
        return content

    # Normal Chat handler
    @commands.command(name="chat")
    async def chat_command(self, message, message_content) -> None:
        if message.guild:
            server_name = message.channel.name
        else:
            server_name = message.author.name
        chatlog_filename = os.path.join(self.chatlog_dir, f"{self.chatbot.char_name}_{server_name}_chatlog.log")
        if ((message.guild and self.chatbot.convo_filename != chatlog_filename) or
            (not message.guild and self.chatbot.convo_filename != chatlog_filename)
        ):
            await self.chatbot.set_convo_filename(chatlog_filename)
        response = await self.chatbot.save_conversation(message, await self.replace_user_mentions(message_content))
        return response

    @app_commands.command( name="followup", description="Make the bot send another message")
    async def followup(self, interaction: discord.Interaction) -> None:
        if interaction.guild:
            server_name = interaction.channel.name
        else:
            server_name = interaction.author.name
        chatlog_filename = os.path.join( self.chatlog_dir, f"{self.chatbot.char_name}_{server_name}_chatlog.log")
        if ((interaction.guild and self.chatbot.convo_filename != chatlog_filename) or
            (not interaction.guild and self.chatbot.convo_filename != chatlog_filename)
        ):
            await self.chatbot.set_convo_filename(chatlog_filename)
        await interaction.response.defer()
        await interaction.delete_original_response()
        await interaction.channel.send(await self.chatbot.follow_up())

    """

    I can't be arsed to deal with this function right now

        @app_commands.command(name="regenerate", description="regenerate last message")
        async def regenerate(self, interaction: discord.Interaction) -> None:
            await interaction.response.defer()
            await interaction.delete_original_response()
            if interaction.guild:
                server_name = interaction.channel.name
            else:
                server_name = interaction.author.name
            chatlog_filename = os.path.join( self.chatlog_dir, f"{self.chatbot.char_name}_{server_name}_chatlog.log")
            if ((interaction.guild and self.chatbot.convo_filename != chatlog_filename) or
                (not interaction.guild and self.chatbot.convo_filename != chatlog_filename)
            ):
                await self.chatbot.set_convo_filename(chatlog_filename)
            # Get the last message sent by the bot in the channel
            async for message in interaction.channel.history(limit=1):
                if message.author == self.bot.user:
                    await message.delete()
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].startswith(f"{self.chatbot.char_name}:"):
                            self.chatbot.remove_item_at_index_and_update_token_count(i)
                            break
                    break  # Exit the loop after deleting the message
            with open(self.chatbot.convo_filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Find the last line that matches "self.chatbot.char_name: {message.content}"
                last_line_num_to_overwrite = None
                for i in range(len(lines) - 1, -1, -1):
                    if f"{self.chatbot.char_name}: {message.content}" in lines[i]:
                        last_line_num_to_overwrite = i
                        break
                if last_line_num_to_overwrite is not None:
                    lines[last_line_num_to_overwrite] = ""
                    # Modify the last line that matches "self.chatbot.char_name: {message.content}"
                with open(self.chatbot.convo_filename, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                    f.close()
            await interaction.channel.send(await self.chatbot.follow_up())
    """

    async def api_get(self, parameter):
        response = requests.get(f"{self.chatbot.endpoint}/api/v1/config/{parameter}")
        return response.json()

    async def api_put(self, parameter, value):
        response = requests.put(f"{self.chatbot.endpoint}/api/v1/config/{parameter}", json={"value": value})
        return response.json()

    @app_commands.command(name="koboldget", description="Get the value of a parameter from the API")
    async def koboldget(self, interaction: discord.Interaction, parameter: str):
        try:
            value = model_config.get(parameter)
            # value = await self.api_get(parameter)
            print(f"Parameter '{parameter}' value: {value}")
            await interaction.response.send_message(
                embed=embedder(f"Parameter {parameter} value: {value}"), delete_after=3
            )
        except Exception as e:
            await interaction.response.send_message(embed=embedder(f"Error: {e}"), delete_after=12)

    @app_commands.command(name="koboldput", description="Set the value of a parameter in the API")
    async def koboldput(self, interaction: discord.Interaction, parameter: str, value: str):
        try:
            model_config[parameter] = float(value)
            await interaction.response.send_message(
                embed=embedder(f"Parameter '{parameter}' updated to: {value}"),
                delete_after=3,
            )
        except Exception as e:
            await interaction.response.send_message(embed=embedder(f"Error: {e}"), delete_after=12)

    @app_commands.command(name="reset_conversation", description="Reset conversation")
    async def reset_conversation(self, interaction: discord.Interaction):
        deleleted_successfully = await self.chatbot.reset_convo_file()
        if deleleted_successfully is not True:
            await interaction.response.send_message(
                "Conversation has not been reset. Please send an initial message first to instantiate the bot instance."
            )
        else:
            await interaction.response.send_message(
                "Current conversation has been deleted. Context has been wiped up until this point."
            )

    @app_commands.command(name="enable_multiline_response", description="Enable responses with multiple lines")
    async def enable_multiline_response(self, interaction: discord.Interaction):
        try:
            if not model_config["stop_sequence"].contains("\n"):
                model_config["stop_sequence"].append("\n")
            await interaction.response.send_message(
                embed=embedder(f"Multi-line responses enabled"),
                delete_after=3,
            )
        except Exception as e:
            await interaction.response.send_message(embed=embedder(f"Error: {e}"), delete_after=12)
    
    @app_commands.command(name="disable_multiline_response", description="Disable responses with multiple lines")
    async def disable_multiline_response(self, interaction: discord.Interaction):
        try:
            if model_config["stop_sequence"].contains("\n"):
                model_config["stop_sequence"].remove("\n")
            await interaction.response.send_message(
                embed=embedder(f"Multi-line responses disabled"),
                delete_after=3,
            )
        except Exception as e:
            await interaction.response.send_message(embed=embedder(f"Error: {e}"), delete_after=12)


async def setup(bot):
    # add chatbot cog to bot
    await bot.add_cog(ChatbotCog(bot))
