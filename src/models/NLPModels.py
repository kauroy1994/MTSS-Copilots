import os
from openai import OpenAI

class LLM:

    def __init__(self,LLM_config):

        self.config = LLM_config

    def execute_prompt(self, prompt="Introduce yourself"):

        return prompt
