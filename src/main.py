import os
from random import choice
from assets.DataUtils import DataLoader
from copilots.Memory_Utils import Knowledge_Representation

class MTSS_QA:

    @staticmethod
    def run_demo():

        user_role = choice(list(DataLoader.system_templates.keys()))
        system_role = choice(list(DataLoader.system_templates.keys()))

        if not user_role is system_role:
            print ("role mismatch ... ")
            return

        mtss_text_data = DataLoader.read_data()
        mtss_data_repr = Knowledge_Representation.create(mtss_text_data)
        print (mtss_data_repr)

if __name__ == '__main__':
    try:
        MTSS_QA.run_demo()
        os.system('rm -rf src/assets/__pycache__')

    except RuntimeError as e:
        print (e)
        os.system('rm -rf src/assets/__pycache__')

    except KeyboardInterrupt:
        os.system('rm -rf src/assets/__pycache__')