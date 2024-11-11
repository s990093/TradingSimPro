import openai
import random

api_keys = ["sk-JsX1k5W4cJmkII0F005273A2E4D1430eBf1bB38a92Dc33Fb",
       "sk-ouapzZyTJaNSUAO480744a8693F446B1B1A552Da73A05971",
       "sk-IjlK3saoqmSqp22f3842B88e8fCd44F0AfD2D8E841C9FaF7"
       ]

openai.base_url = "https://free.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}

def analyze_with_gpt(content, task_prompt, model="gpt-3.5-turbo-0125", temperature=0, max_tokens=200, top_p=1, frequency_penalty=0.0, presence_penalty=0.0):
    """
    This function uses the OpenAI GPT model to generate a response based on the given content and task prompt.

    Parameters:
    - content (str): The input text content for the GPT model.
    - task_prompt (str): The task prompt that guides the GPT model's response.
    - model (str): The GPT model to be used. Default is "gpt-3.5-turbo-0125".
    - temperature (float): The randomness of the GPT model's output. Default is 0.
    - max_tokens (int): The maximum number of tokens to generate. Default is 200.
    - top_p (float): The nucleus sampling parameter. Default is 1.
    - frequency_penalty (float): The penalty for repeating the same words. Default is 0.0.
    - presence_penalty (float): The penalty for repeating the same phrases. Default is 0.0.

    Returns:
    - str: The generated response from the GPT model. If an error occurs, returns "Error in generating response".
    """
    
    openai.api_key = random.choice(api_keys)

    try:
        completion = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=[
                {"role": "system", "content": task_prompt},
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        # print("Traceback:")
        # traceback.print_exc()  # 打印錯誤堆棧
        # print( {"role": "system", "content": task_prompt},
        #         {"role": "user", "content": content})
        return "Error in generating response"