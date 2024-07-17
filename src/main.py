import os
from random import choice
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM

class MTSS_Copilot:

    @staticmethod
    def simulate_user_turn():

        #initialize user
        users_and_queries = AssetLoader.get_queries()
        user_roles = list(users_and_queries.keys())

        random_user_role = choice(user_roles)
        random_user_query = choice(users_and_queries[random_user_role])
        return random_user_role, random_user_query

    @staticmethod
    def simulate_QA_agent_turn(user_role, user_query, data):

        llm_response = None
        context = Retr.retrieve_context(data,user_query,symb_model=Symbolic_Model(),top_k=1)[0]
        system_template = AssetLoader.get_templates()[user_role]
        llm = LLM()
        llm.set_prompt(system_template,user_query,context)
        llm_response = llm.respond_to_prompt()
        return system_template, llm_response

    @staticmethod
    def simulate_summary_agent(total_information):

        llm_response = None
        llm = LLM()
        llm.set_prompt(system_template="summary template",content=total_information)
        llm_response = llm.respond_to_prompt()
        return llm_response

    @staticmethod
    def run_demo(turns = 2):
        
        mtss_text_data = AssetLoader.read_data()
        mtss_data_repr = Knowledge_Representation.organize_data(mtss_text_data)

        total_information = ""

        for _ in range(turns):
            user_role, user_query = MTSS_Copilot.simulate_user_turn()

            print ('\n ===== USER attributes =====\n')
            print ('user role:', user_role)
            print ('user_query', user_query)
            
            agent_instructions, agent_response = MTSS_Copilot.simulate_QA_agent_turn(user_role, user_query, mtss_data_repr)
            total_information += str(agent_response)
            print ('\n ===== SYSTEM INSTRUCTIONS ===== \n',agent_instructions)
            print ('\n ===== SYSTEM RESPONSE ===== \n',agent_response)

        summary_agent_response = MTSS_Copilot.simulate_summary_agent(total_information)
        print ('\n ===== INFORMATION SUMMARY ===== \n',summary_agent_response)

if __name__ == '__main__':
    try:
        MTSS_Copilot.run_demo()
        os.system('rm -rf src/assets/__pycache__')
        os.system('rm -rf src/copilots/__pycache__')

    except RuntimeError as e:
        print (e)
        os.system('rm -rf src/assets/__pycache__')
        os.system('rm -rf src/copilots/__pycache__')

    except KeyboardInterrupt:
        os.system('rm -rf src/assets/__pycache__')
        os.system('rm -rf src/copilots/__pycache__')