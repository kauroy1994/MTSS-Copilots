import os
from assets.DataUtils import DataLoader

class MTSS_QA:

    @staticmethod
    def run_demo():

        mtss_text_data = DataLoader.read_data()
        print (len(mtss_text_data)); input()

if __name__ == '__main__':
    try:
        MTSS_QA.run_demo()
        os.system('rm -rf src/assets/__pycache__')

    except RuntimeError as e:
        print (e)
        os.system('rm -rf src/assets/__pycache__')

    except KeyboardInterrupt:
        os.system('rm -rf src/assets/__pycache__')