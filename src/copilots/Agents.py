from decouple import config
from groq import Groq

class LLM:

    def __init__(self,api='GROQ',groq_model="mixtral-8x7b-32768"):

        if api == 'GROQ':
            api_key = config('GROQ_API_KEY')
            self.groq_client = Groq(api_key=api_key)
            self.groq_model = groq_model
    
    def prompt_llm(self,prompt,max_retries=5):
        """
        returns llm response based on prompt
        """

        try:

            client = self.groq_client
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        }
                        ],
                        temperature = 0.0,
                        model=self.groq_model,
                        )
             

        except Exception:
            print ("Unsupported LLM api")
            exit()
    