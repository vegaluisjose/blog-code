import json
from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI()

    model = "gpt-4o"

    # define tool for weather
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # temperatures to be tested
    temperatures = [i * 10 for i in range(11)]
    city = "Boston"

    # let's see how model responds under different temperatures
    for temp in temperatures:
        # new conversation for every temperature
        messages = [
            {
                "role": "user",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"What's the weather like in {city} today?",
            },
        ]

        # create the first completion
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )

        # get tool_calls from completion
        tool_calls = completion.choices[0].message.tool_calls

        # if model requires to call a function
        if tool_calls:
            # add tool call messages
            messages.append(
                {
                    "role": completion.choices[0].message.role,
                    "content": "",
                    "tool_calls": [
                        tool_call.model_dump()
                        for tool_call in completion.choices[0].message.tool_calls
                    ],
                }
            )

            # get first function since we are asking for one location only
            function_name = tool_calls[0].function.name

            # get the arguments
            function_args = json.loads(tool_calls[0].function.arguments)

            # language enriched response
            # tool_response = f"The temperature in {city} today is {temp} fahrenheit."

            # JSON enriched response
            tool_response = json.dumps(
                {"city": city, "temperature": temp, "unit": "fahrenheit"}
            )

            # just a numeric response
            # tool_response = f"{temp}"

            # add tool response to messages
            messages.append(
                {
                    "tool_call_id": tool_calls[0].id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_response,
                }
            )

            # run a completion with the call and response added to the conversation
            last_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )

            # print results
            print(
                f"Tool response:{tool_response}\nMessage:{last_completion.choices[0].message.content}\n"
            )

