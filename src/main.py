import json, os
import gradio as gr
from models.NLPModels import LLM
from miscellaneous.PrintUtils import Format
from miscellaneous.FileIO import JSON
from assets.DataLoader import MTSS_data_loader
from vector_database.Implementations import Faiss

class Copilot:

    @staticmethod
    def run(config_json, test_query = "What is MTSS?", role=None):

        data = MTSS_data_loader.get_data_from_folder_path(config_json["data_folder_path"])
        database = Faiss.prepare_data(data)
        Faiss.create_index(database)
        test_query = "What is MTSS?"
        score, response1 = Faiss.search(test_query,database)

        if not role is None:
            data = MTSS_data_loader.get_data_from_folder_path(config_json["data_folder_path"],role=role)
            database = Faiss.prepare_data(data)
            Faiss.create_index(database)
            test_query = "What is MTSS?"
            score, response2 = Faiss.search(test_query,database)
            return ('\n'+Format.green(response1)+'\n\n'+Format.green(response2))
        else:
            return ('\n'+Format.green(response1))

if __name__ == '__main__':

    GRADIO = False

    config_json = JSON.read_from_path('config.json')
    #roles: ['admin','clinical_staff','school_counselor','school_psych','teachers']
    response = Copilot.run(config_json,test_query = "What is MTSS?", role='admin')
    print (response)

    if GRADIO:
        def response(test_query, role):
            return Copilot.run(config_json,test_query = test_query, role=role)

        gr.ChatInterface(response(test_query = "What is MTSS?", role='admin')).launch()
