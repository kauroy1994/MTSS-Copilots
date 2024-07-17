import os
from random import choice
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation
from copilots.Agents import LLM

class MTSS_Copilot:

    @staticmethod
    def simulate_user_query():

        #initialize user
        users_and_queries = AssetLoader.get_queries()
        user_roles = list(users_and_queries.keys())

        random_user_role = choice(user_roles)
        random_user_query = choice(users_and_queries[random_user_role])
        return random_user_role, random_user_query

    @staticmethod
    def run_demo():
        
        #format and organize data -- grounding/alignment w customization
        mtss_text_data = AssetLoader.read_data()
        mtss_data_repr = Knowledge_Representation.organize_data(mtss_text_data)
        user_role, user_query = MTSS_Copilot.simulate_user_query()
        print (user_role, user_query)
        #-- alignment/instructability w customization


if __name__ == '__main__':
    try:
        MTSS_Copilot.run_demo()
        os.system('rm -rf src/assets/__pycache__')

    except RuntimeError as e:
        print (e)
        os.system('rm -rf src/assets/__pycache__')

    except KeyboardInterrupt:
        os.system('rm -rf src/assets/__pycache__')